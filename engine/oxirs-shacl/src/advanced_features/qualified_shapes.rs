//! SHACL Advanced Features - Qualified Value Shapes
//!
//! Implementation of qualified value shape constraints including:
//! - sh:qualifiedValueShape - constrains values based on their conformance to a shape
//! - sh:qualifiedMinCount - minimum number of values conforming to the qualified shape
//! - sh:qualifiedMaxCount - maximum number of values conforming to the qualified shape
//! - sh:qualifiedValueShapesDisjoint - ensures qualified shapes are disjoint
//!
//! Based on the W3C SHACL specification for qualified value shapes.

use crate::{
    validation::ValidationEngine, PropertyPath, Result, ShaclError, Shape, ShapeId,
    ValidationConfig,
};
use oxirs_core::{model::Term, Store};
use serde::{Deserialize, Serialize};

/// Qualified value shape constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualifiedValueShapeConstraint {
    /// The shape that values must conform to
    pub qualified_shape: QualifiedShape,
    /// Minimum count of values that must conform (inclusive)
    pub min_count: Option<usize>,
    /// Maximum count of values that may conform (inclusive)
    pub max_count: Option<usize>,
    /// Whether qualified shapes must be disjoint
    pub disjoint: bool,
}

/// A qualified shape reference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QualifiedShape {
    /// Reference to a named shape by ID
    ShapeRef(ShapeId),
    /// Inline shape definition
    InlineShape(Box<Shape>),
}

impl PartialEq for QualifiedShape {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (QualifiedShape::ShapeRef(id1), QualifiedShape::ShapeRef(id2)) => id1 == id2,
            (QualifiedShape::InlineShape(s1), QualifiedShape::InlineShape(s2)) => {
                // Compare shape IDs for inline shapes
                s1.id == s2.id
            }
            _ => false,
        }
    }
}

impl Eq for QualifiedShape {}

impl QualifiedValueShapeConstraint {
    /// Create a new qualified value shape constraint
    pub fn new(qualified_shape: QualifiedShape) -> Self {
        Self {
            qualified_shape,
            min_count: None,
            max_count: None,
            disjoint: false,
        }
    }

    /// Set minimum count
    pub fn with_min_count(mut self, min_count: usize) -> Self {
        self.min_count = Some(min_count);
        self
    }

    /// Set maximum count
    pub fn with_max_count(mut self, max_count: usize) -> Self {
        self.max_count = Some(max_count);
        self
    }

    /// Set disjoint flag
    pub fn with_disjoint(mut self, disjoint: bool) -> Self {
        self.disjoint = disjoint;
        self
    }

    /// Validate this constraint
    pub fn validate(
        &self,
        focus_node: &Term,
        path: Option<&PropertyPath>,
        values: &[Term],
        store: &dyn Store,
        shape_registry: &dyn ShapeRegistry,
    ) -> Result<QualifiedValidationResult> {
        // Get the qualified shape
        let shape = match &self.qualified_shape {
            QualifiedShape::ShapeRef(shape_id) => shape_registry
                .get_shape(shape_id)
                .ok_or_else(|| {
                    ShaclError::ShapeValidation(format!("Qualified shape not found: {}", shape_id))
                })?
                .clone(),
            QualifiedShape::InlineShape(shape) => (**shape).clone(),
        };

        // Count how many values conform to the qualified shape
        let mut conforming_values = Vec::new();
        let mut non_conforming_values = Vec::new();

        for value in values {
            let conforms = self.value_conforms_to_shape(value, &shape, store, shape_registry)?;

            if conforms {
                conforming_values.push(value.clone());
            } else {
                non_conforming_values.push(value.clone());
            }
        }

        let conforming_count = conforming_values.len();

        // Check minimum count constraint
        if let Some(min_count) = self.min_count {
            if conforming_count < min_count {
                return Ok(QualifiedValidationResult::violation(
                    format!(
                        "Qualified value shape requires at least {} conforming values, found {}",
                        min_count, conforming_count
                    ),
                    conforming_values,
                    non_conforming_values,
                ));
            }
        }

        // Check maximum count constraint
        if let Some(max_count) = self.max_count {
            if conforming_count > max_count {
                return Ok(QualifiedValidationResult::violation(
                    format!(
                        "Qualified value shape allows at most {} conforming values, found {}",
                        max_count, conforming_count
                    ),
                    conforming_values,
                    non_conforming_values,
                ));
            }
        }

        Ok(QualifiedValidationResult::success(
            conforming_values,
            non_conforming_values,
        ))
    }

    /// Check whether a value node conforms to the qualified shape.
    ///
    /// The value is validated against `shape` using a [`ValidationEngine`]. So
    /// that any shapes referenced from `shape` (via `sh:node`, `sh:property`,
    /// nested qualified shapes, ...) resolve, every named shape obtainable from
    /// `shape_registry` is copied into the temporary shapes map alongside the
    /// qualified shape itself. The value conforms iff the produced report
    /// conforms.
    fn value_conforms_to_shape(
        &self,
        value: &Term,
        shape: &Shape,
        store: &dyn Store,
        shape_registry: &dyn ShapeRegistry,
    ) -> Result<bool> {
        tracing::debug!("Checking conformance of {:?} to shape {}", value, shape.id);

        let mut temp_shapes = indexmap::IndexMap::new();
        // Pull in every named shape we can resolve so nested references work.
        // `ShapeRegistry` does not enumerate shapes, but the qualified shape may
        // itself reference others by id; those are resolved on demand below.
        temp_shapes.insert(shape.id.clone(), shape.clone());
        for referenced in collect_referenced_shape_ids(shape) {
            if let Some(referenced_shape) = shape_registry.get_shape(&referenced) {
                temp_shapes
                    .entry(referenced)
                    .or_insert_with(|| referenced_shape.clone());
            }
        }

        let config = ValidationConfig::default();
        let mut validator = ValidationEngine::new(&temp_shapes, config);

        match validator.validate_node_against_shape(store, shape, value, None) {
            Ok(report) => Ok(report.conforms()),
            Err(e) => {
                tracing::warn!("Qualified value shape validation error: {e}");
                Ok(false)
            }
        }
    }
}

/// Collect the IDs of every shape referenced from `shape`'s constraints,
/// property shapes, and `sh:extends` parents.
///
/// Used to populate the temporary shapes map for nested validation so that
/// references such as `sh:node`, `sh:property`, `sh:and`/`sh:or`/`sh:xone`,
/// `sh:not`, and nested `sh:qualifiedValueShape` resolve correctly.
fn collect_referenced_shape_ids(shape: &Shape) -> Vec<ShapeId> {
    use crate::Constraint;

    let mut ids = Vec::new();
    ids.extend(shape.property_shapes.iter().cloned());
    ids.extend(shape.extends.iter().cloned());

    for constraint in shape.constraints.values() {
        match constraint {
            Constraint::Node(c) => ids.push(c.shape.clone()),
            Constraint::Not(c) => ids.push(c.shape.clone()),
            Constraint::And(c) => ids.extend(c.shapes.iter().cloned()),
            Constraint::Or(c) => ids.extend(c.shapes.iter().cloned()),
            Constraint::Xone(c) => ids.extend(c.shapes.iter().cloned()),
            Constraint::QualifiedValueShape(c) => ids.push(c.shape.clone()),
            _ => {}
        }
    }

    ids
}

/// Result of qualified value shape validation
#[derive(Debug, Clone)]
pub struct QualifiedValidationResult {
    /// Whether validation passed
    pub conforms: bool,
    /// Error message if validation failed
    pub message: Option<String>,
    /// Values that conformed to the qualified shape
    pub conforming_values: Vec<Term>,
    /// Values that did not conform
    pub non_conforming_values: Vec<Term>,
}

impl QualifiedValidationResult {
    /// Create a success result
    pub fn success(conforming_values: Vec<Term>, non_conforming_values: Vec<Term>) -> Self {
        Self {
            conforms: true,
            message: None,
            conforming_values,
            non_conforming_values,
        }
    }

    /// Create a violation result
    pub fn violation(
        message: String,
        conforming_values: Vec<Term>,
        non_conforming_values: Vec<Term>,
    ) -> Self {
        Self {
            conforms: false,
            message: Some(message),
            conforming_values,
            non_conforming_values,
        }
    }

    /// Check if validation passed
    pub fn is_valid(&self) -> bool {
        self.conforms
    }

    /// Get conforming count
    pub fn conforming_count(&self) -> usize {
        self.conforming_values.len()
    }
}

/// Trait for accessing shapes in qualified value shape validation
pub trait ShapeRegistry {
    /// Get a shape by ID
    fn get_shape(&self, id: &ShapeId) -> Option<&Shape>;

    /// Check if shapes are disjoint
    fn are_shapes_disjoint(&self, shape1: &ShapeId, shape2: &ShapeId) -> Result<bool>;
}

/// Qualified shapes validator with disjointness checking
pub struct QualifiedShapesValidator {
    /// Cache for disjointness checks
    disjointness_cache: std::collections::HashMap<(ShapeId, ShapeId), bool>,
}

impl QualifiedShapesValidator {
    /// Create a new validator
    pub fn new() -> Self {
        Self {
            disjointness_cache: std::collections::HashMap::new(),
        }
    }

    /// Validate multiple qualified value shapes with disjointness checking
    pub fn validate_with_disjointness(
        &mut self,
        constraints: &[QualifiedValueShapeConstraint],
        focus_node: &Term,
        path: Option<&PropertyPath>,
        values: &[Term],
        store: &dyn Store,
        shape_registry: &dyn ShapeRegistry,
    ) -> Result<Vec<QualifiedValidationResult>> {
        // First, check if any constraints require disjoint shapes
        let has_disjoint_requirement = constraints.iter().any(|c| c.disjoint);

        if has_disjoint_requirement {
            // Verify that all qualified shapes are pairwise disjoint
            self.verify_disjointness(constraints, shape_registry)?;
        }

        // Validate each constraint
        let mut results = Vec::new();
        for constraint in constraints {
            let result = constraint.validate(focus_node, path, values, store, shape_registry)?;
            results.push(result);
        }

        Ok(results)
    }

    /// Verify that all qualified shapes in the constraints are pairwise disjoint
    fn verify_disjointness(
        &mut self,
        constraints: &[QualifiedValueShapeConstraint],
        shape_registry: &dyn ShapeRegistry,
    ) -> Result<()> {
        // Get all shape IDs
        let shape_ids: Vec<ShapeId> = constraints
            .iter()
            .filter_map(|c| match &c.qualified_shape {
                QualifiedShape::ShapeRef(id) => Some(id.clone()),
                QualifiedShape::InlineShape(_) => None, // Skip inline shapes for now
            })
            .collect();

        // Check pairwise disjointness
        for i in 0..shape_ids.len() {
            for j in (i + 1)..shape_ids.len() {
                let shape1 = &shape_ids[i];
                let shape2 = &shape_ids[j];

                // Check cache
                let cache_key = if shape1 < shape2 {
                    (shape1.clone(), shape2.clone())
                } else {
                    (shape2.clone(), shape1.clone())
                };

                let disjoint = if let Some(&cached) = self.disjointness_cache.get(&cache_key) {
                    cached
                } else {
                    let disjoint = shape_registry.are_shapes_disjoint(shape1, shape2)?;
                    self.disjointness_cache.insert(cache_key, disjoint);
                    disjoint
                };

                if !disjoint {
                    return Err(ShaclError::ConstraintValidation(format!(
                        "Qualified shapes {} and {} are not disjoint",
                        shape1, shape2
                    )));
                }
            }
        }

        Ok(())
    }

    /// Clear the disjointness cache
    pub fn clear_cache(&mut self) {
        self.disjointness_cache.clear();
    }
}

impl Default for QualifiedShapesValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Complex qualified value shapes with multiple conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexQualifiedConstraint {
    /// All of these qualified shapes must be satisfied (AND)
    pub all_of: Vec<QualifiedValueShapeConstraint>,
    /// At least one of these qualified shapes must be satisfied (OR)
    pub any_of: Vec<QualifiedValueShapeConstraint>,
    /// None of these qualified shapes should be satisfied (NOT)
    pub none_of: Vec<QualifiedValueShapeConstraint>,
    /// Exactly one of these qualified shapes must be satisfied (XOR)
    pub one_of: Vec<QualifiedValueShapeConstraint>,
}

impl ComplexQualifiedConstraint {
    /// Create a new complex constraint
    pub fn new() -> Self {
        Self {
            all_of: Vec::new(),
            any_of: Vec::new(),
            none_of: Vec::new(),
            one_of: Vec::new(),
        }
    }

    /// Add an ALL OF constraint
    pub fn add_all_of(mut self, constraint: QualifiedValueShapeConstraint) -> Self {
        self.all_of.push(constraint);
        self
    }

    /// Add an ANY OF constraint
    pub fn add_any_of(mut self, constraint: QualifiedValueShapeConstraint) -> Self {
        self.any_of.push(constraint);
        self
    }

    /// Add a NONE OF constraint
    pub fn add_none_of(mut self, constraint: QualifiedValueShapeConstraint) -> Self {
        self.none_of.push(constraint);
        self
    }

    /// Add a ONE OF constraint
    pub fn add_one_of(mut self, constraint: QualifiedValueShapeConstraint) -> Self {
        self.one_of.push(constraint);
        self
    }

    /// Validate this complex constraint
    pub fn validate(
        &self,
        focus_node: &Term,
        path: Option<&PropertyPath>,
        values: &[Term],
        store: &dyn Store,
        shape_registry: &dyn ShapeRegistry,
    ) -> Result<ComplexValidationResult> {
        let mut errors = Vec::new();

        // Validate ALL OF constraints
        for constraint in &self.all_of {
            let result = constraint.validate(focus_node, path, values, store, shape_registry)?;
            if !result.is_valid() {
                errors.push(format!(
                    "ALL OF constraint failed: {}",
                    result.message.unwrap_or_default()
                ));
            }
        }

        // Validate ANY OF constraints (at least one must pass)
        if !self.any_of.is_empty() {
            let any_passed = self.any_of.iter().any(|c| {
                c.validate(focus_node, path, values, store, shape_registry)
                    .ok()
                    .and_then(|r| if r.is_valid() { Some(()) } else { None })
                    .is_some()
            });

            if !any_passed {
                errors.push("None of the ANY OF constraints were satisfied".to_string());
            }
        }

        // Validate NONE OF constraints (all must fail)
        for constraint in &self.none_of {
            let result = constraint.validate(focus_node, path, values, store, shape_registry)?;
            if result.is_valid() {
                errors.push("NONE OF constraint unexpectedly passed".to_string());
            }
        }

        // Validate ONE OF constraints (exactly one must pass)
        if !self.one_of.is_empty() {
            let passed_count = self
                .one_of
                .iter()
                .filter(|c| {
                    c.validate(focus_node, path, values, store, shape_registry)
                        .ok()
                        .and_then(|r| if r.is_valid() { Some(()) } else { None })
                        .is_some()
                })
                .count();

            if passed_count != 1 {
                errors.push(format!(
                    "ONE OF constraint requires exactly 1 match, found {}",
                    passed_count
                ));
            }
        }

        if errors.is_empty() {
            Ok(ComplexValidationResult::success())
        } else {
            Ok(ComplexValidationResult::violation(errors))
        }
    }
}

impl Default for ComplexQualifiedConstraint {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of complex qualified constraint validation
#[derive(Debug, Clone)]
pub struct ComplexValidationResult {
    /// Whether validation passed
    pub conforms: bool,
    /// Error messages if validation failed
    pub errors: Vec<String>,
}

impl ComplexValidationResult {
    /// Create a success result
    pub fn success() -> Self {
        Self {
            conforms: true,
            errors: Vec::new(),
        }
    }

    /// Create a violation result
    pub fn violation(errors: Vec<String>) -> Self {
        Self {
            conforms: false,
            errors,
        }
    }

    /// Check if validation passed
    pub fn is_valid(&self) -> bool {
        self.conforms
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::value_constraints::ClassConstraint;
    use crate::{Constraint, ConstraintComponentId, ShapeType};
    use oxirs_core::{
        model::{GraphName, NamedNode, Object, Predicate, Quad, Subject},
        ConcreteStore,
    };
    use std::collections::HashMap;

    const EX: &str = "http://example.org/";
    const RDF_TYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";

    fn iri(local: &str) -> NamedNode {
        NamedNode::new(format!("{EX}{local}")).expect("valid IRI")
    }

    fn term(local: &str) -> Term {
        Term::NamedNode(iri(local))
    }

    fn insert_type(store: &ConcreteStore, subject: &str, type_local: &str) {
        let quad = Quad::new(
            Subject::from(iri(subject)),
            Predicate::from(NamedNode::new(RDF_TYPE).expect("rdf:type")),
            Object::from(iri(type_local)),
            GraphName::DefaultGraph,
        );
        store.insert_quad(quad).expect("insert type triple");
    }

    /// Minimal in-memory shape registry backed by a `HashMap`.
    struct MapRegistry {
        shapes: HashMap<ShapeId, Shape>,
    }

    impl MapRegistry {
        fn new() -> Self {
            Self {
                shapes: HashMap::new(),
            }
        }

        fn with(mut self, shape: Shape) -> Self {
            self.shapes.insert(shape.id.clone(), shape);
            self
        }
    }

    impl ShapeRegistry for MapRegistry {
        fn get_shape(&self, id: &ShapeId) -> Option<&Shape> {
            self.shapes.get(id)
        }

        fn are_shapes_disjoint(&self, _shape1: &ShapeId, _shape2: &ShapeId) -> Result<bool> {
            Ok(true)
        }
    }

    /// Node shape requiring `sh:class :Friend`.
    fn friend_shape(id: &str) -> Shape {
        let mut shape = Shape::node_shape(ShapeId::new(id));
        shape.add_constraint(
            ConstraintComponentId::new("sh:ClassConstraintComponent"),
            Constraint::Class(ClassConstraint {
                class_iri: iri("Friend"),
            }),
        );
        shape
    }

    #[test]
    fn test_qualified_counts_exact_conforming() {
        // Two friends + one stranger; only friends conform to the qualified shape.
        let store = ConcreteStore::new().expect("store");
        insert_type(&store, "f1", "Friend");
        insert_type(&store, "f2", "Friend");
        // s1 has no rdf:type Friend => non-conforming.

        let registry = MapRegistry::new().with(friend_shape("FriendShape"));
        let constraint = QualifiedValueShapeConstraint::new(QualifiedShape::ShapeRef(
            ShapeId::new("FriendShape"),
        ));

        let values = vec![term("f1"), term("f2"), term("s1")];
        let result = constraint
            .validate(&term("focus"), None, &values, &store, &registry)
            .expect("validate");

        assert_eq!(
            result.conforming_count(),
            2,
            "exactly two values conform to FriendShape"
        );
        assert_eq!(result.non_conforming_values.len(), 1);
        assert!(result.is_valid(), "no min/max set => success");
    }

    #[test]
    fn test_qualified_min_count_satisfied() {
        let store = ConcreteStore::new().expect("store");
        insert_type(&store, "f1", "Friend");
        insert_type(&store, "f2", "Friend");

        let registry = MapRegistry::new().with(friend_shape("FriendShape"));
        let constraint = QualifiedValueShapeConstraint::new(QualifiedShape::ShapeRef(
            ShapeId::new("FriendShape"),
        ))
        .with_min_count(2);

        let values = vec![term("f1"), term("f2")];
        let result = constraint
            .validate(&term("focus"), None, &values, &store, &registry)
            .expect("validate");
        assert!(result.is_valid(), "2 conforming >= min 2 => valid");
    }

    #[test]
    fn test_qualified_min_count_violated() {
        let store = ConcreteStore::new().expect("store");
        insert_type(&store, "f1", "Friend");
        // Only one friend; the rest are strangers.

        let registry = MapRegistry::new().with(friend_shape("FriendShape"));
        let constraint = QualifiedValueShapeConstraint::new(QualifiedShape::ShapeRef(
            ShapeId::new("FriendShape"),
        ))
        .with_min_count(2);

        let values = vec![term("f1"), term("s1"), term("s2")];
        let result = constraint
            .validate(&term("focus"), None, &values, &store, &registry)
            .expect("validate");
        assert!(!result.is_valid(), "1 conforming < min 2 => violation");
        assert_eq!(result.conforming_count(), 1);
    }

    #[test]
    fn test_qualified_max_count_violated() {
        let store = ConcreteStore::new().expect("store");
        insert_type(&store, "f1", "Friend");
        insert_type(&store, "f2", "Friend");
        insert_type(&store, "f3", "Friend");

        let registry = MapRegistry::new().with(friend_shape("FriendShape"));
        let constraint = QualifiedValueShapeConstraint::new(QualifiedShape::ShapeRef(
            ShapeId::new("FriendShape"),
        ))
        .with_max_count(2);

        let values = vec![term("f1"), term("f2"), term("f3")];
        let result = constraint
            .validate(&term("focus"), None, &values, &store, &registry)
            .expect("validate");
        assert!(!result.is_valid(), "3 conforming > max 2 => violation");
        assert_eq!(result.conforming_count(), 3);
    }

    #[test]
    fn test_qualified_min_max_window_ok() {
        let store = ConcreteStore::new().expect("store");
        insert_type(&store, "f1", "Friend");
        insert_type(&store, "f2", "Friend");

        let registry = MapRegistry::new().with(friend_shape("FriendShape"));
        let constraint = QualifiedValueShapeConstraint::new(QualifiedShape::ShapeRef(
            ShapeId::new("FriendShape"),
        ))
        .with_min_count(1)
        .with_max_count(3);

        let values = vec![term("f1"), term("f2"), term("s1")];
        let result = constraint
            .validate(&term("focus"), None, &values, &store, &registry)
            .expect("validate");
        assert!(result.is_valid(), "2 conforming within [1,3] => valid");
        assert_eq!(result.conforming_count(), 2);
    }

    #[test]
    fn test_qualified_inline_shape_counts() {
        // Use an inline (boxed) shape rather than a registry reference.
        let store = ConcreteStore::new().expect("store");
        insert_type(&store, "f1", "Friend");

        let registry = MapRegistry::new();
        let inline = friend_shape("InlineFriend");
        let constraint =
            QualifiedValueShapeConstraint::new(QualifiedShape::InlineShape(Box::new(inline)))
                .with_min_count(1);

        let values = vec![term("f1"), term("s1")];
        let result = constraint
            .validate(&term("focus"), None, &values, &store, &registry)
            .expect("validate");
        assert_eq!(result.conforming_count(), 1, "inline shape conformance");
        assert!(result.is_valid());
    }

    #[test]
    fn test_collect_referenced_shape_ids_covers_variants() {
        use crate::constraints::logical_constraints::{AndConstraint, NotConstraint};
        use crate::constraints::shape_constraints::NodeConstraint;

        let mut shape = Shape::new(ShapeId::new("root"), ShapeType::NodeShape);
        shape.property_shapes.push(ShapeId::new("propRef"));
        shape.extends.push(ShapeId::new("parent"));
        shape.add_constraint(
            ConstraintComponentId::new("sh:NodeConstraintComponent"),
            Constraint::Node(NodeConstraint::new(ShapeId::new("nodeRef"))),
        );
        shape.add_constraint(
            ConstraintComponentId::new("sh:NotConstraintComponent"),
            Constraint::Not(NotConstraint::new(ShapeId::new("notRef"))),
        );
        shape.add_constraint(
            ConstraintComponentId::new("sh:AndConstraintComponent"),
            Constraint::And(AndConstraint::new(vec![
                ShapeId::new("andRef1"),
                ShapeId::new("andRef2"),
            ])),
        );

        let ids = collect_referenced_shape_ids(&shape);
        for expected in [
            "propRef", "parent", "nodeRef", "notRef", "andRef1", "andRef2",
        ] {
            assert!(
                ids.contains(&ShapeId::new(expected)),
                "missing referenced shape id: {expected}"
            );
        }
    }

    #[test]
    fn test_qualified_constraint_creation() {
        let shape_id = ShapeId::new("test:shape");
        let constraint = QualifiedValueShapeConstraint::new(QualifiedShape::ShapeRef(shape_id))
            .with_min_count(1)
            .with_max_count(5);

        assert_eq!(constraint.min_count, Some(1));
        assert_eq!(constraint.max_count, Some(5));
    }

    #[test]
    fn test_complex_constraint_creation() {
        let shape_id = ShapeId::new("test:shape");
        let qualified = QualifiedValueShapeConstraint::new(QualifiedShape::ShapeRef(shape_id));

        let complex = ComplexQualifiedConstraint::new()
            .add_all_of(qualified.clone())
            .add_any_of(qualified);

        assert_eq!(complex.all_of.len(), 1);
        assert_eq!(complex.any_of.len(), 1);
    }

    #[test]
    fn test_validation_result() {
        let result = QualifiedValidationResult::success(vec![], vec![]);
        assert!(result.is_valid());
        assert_eq!(result.conforming_count(), 0);
    }

    #[test]
    fn test_complex_validation_result() {
        let result = ComplexValidationResult::success();
        assert!(result.is_valid());
        assert!(result.errors.is_empty());

        let result = ComplexValidationResult::violation(vec!["error".to_string()]);
        assert!(!result.is_valid());
        assert_eq!(result.errors.len(), 1);
    }
}
