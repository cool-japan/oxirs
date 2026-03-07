//! SHACL `sh:node` constraint component.
//!
//! The `sh:node` constraint requires that a focus node also conforms to one or
//! more referenced SHACL shapes. This module provides an in-memory
//! representation and a simple validator backed by a conforming-shape registry.

use std::collections::HashSet;

/// A reference to another SHACL shape, identified by its IRI or blank-node
/// label.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ShapeRef {
    /// The IRI or blank-node label of the referenced shape.
    pub shape_iri: String,
}

impl ShapeRef {
    /// Creates a new `ShapeRef` from the given IRI string.
    pub fn new(shape_iri: impl Into<String>) -> Self {
        Self {
            shape_iri: shape_iri.into(),
        }
    }
}

/// A SHACL `sh:node` constraint component.
///
/// A focus node is valid under this constraint only if it conforms to **all**
/// shapes listed in `shape_refs`.
#[derive(Debug, Clone, Default)]
pub struct NodeConstraint {
    /// The set of shape references that the focus node must conform to.
    pub shape_refs: Vec<ShapeRef>,
}

impl NodeConstraint {
    /// Creates a new, empty `NodeConstraint`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a shape reference (by IRI) to the constraint, using a builder
    /// pattern.
    pub fn add_shape(mut self, shape_iri: &str) -> Self {
        self.shape_refs.push(ShapeRef::new(shape_iri));
        self
    }

    /// Returns the number of shape references in this constraint.
    pub fn shape_count(&self) -> usize {
        self.shape_refs.len()
    }

    /// Returns `true` if this constraint references the given shape IRI.
    pub fn contains(&self, shape_iri: &str) -> bool {
        self.shape_refs.iter().any(|r| r.shape_iri == shape_iri)
    }

    /// Returns `true` if the constraint has no shape references.
    pub fn is_empty(&self) -> bool {
        self.shape_refs.is_empty()
    }

    /// Returns an iterator over all shape references.
    pub fn iter(&self) -> impl Iterator<Item = &ShapeRef> {
        self.shape_refs.iter()
    }
}

/// The outcome of validating a single node against a single shape reference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NodeValidationResult {
    /// The focus node conforms to the referenced shape.
    Conforms,
    /// The focus node violates the referenced shape constraint.
    Violation {
        /// The IRI of the shape that was violated.
        shape_iri: String,
        /// A human-readable message describing the violation.
        message: String,
    },
}

impl NodeValidationResult {
    /// Returns `true` if this result indicates conformance.
    pub fn is_conforms(&self) -> bool {
        matches!(self, NodeValidationResult::Conforms)
    }

    /// Returns `true` if this result indicates a violation.
    pub fn is_violation(&self) -> bool {
        matches!(self, NodeValidationResult::Violation { .. })
    }

    /// Returns the shape IRI from a violation result, or `None` if it conforms.
    pub fn violation_shape(&self) -> Option<&str> {
        if let NodeValidationResult::Violation { shape_iri, .. } = self {
            Some(shape_iri.as_str())
        } else {
            None
        }
    }
}

/// Validates node constraints against an in-memory registry of conforming shapes.
///
/// The validator maintains a set of shape IRIs that are assumed to be "known
/// valid" (i.e., any focus node is considered to conform to those shapes).
/// Shape IRIs not present in the registry are treated as violations.
pub struct NodeConstraintValidator {
    /// Set of shape IRIs for which any node is considered conforming.
    conforming_shapes: HashSet<String>,
}

impl NodeConstraintValidator {
    /// Creates a new validator with an empty conforming-shapes registry.
    pub fn new() -> Self {
        Self {
            conforming_shapes: HashSet::new(),
        }
    }

    /// Registers a shape IRI as conforming, using a builder pattern.
    pub fn register_conforming(mut self, shape_iri: &str) -> Self {
        self.conforming_shapes.insert(shape_iri.to_string());
        self
    }

    /// Validates the focus node identified by `node_iri` against every shape
    /// reference in `constraint`.
    ///
    /// Returns one `NodeValidationResult` per shape reference. Shape IRIs
    /// present in the conforming registry produce `Conforms`; all others
    /// produce a `Violation`.
    pub fn validate(
        &self,
        node_iri: &str,
        constraint: &NodeConstraint,
    ) -> Vec<NodeValidationResult> {
        constraint
            .shape_refs
            .iter()
            .map(|shape_ref| {
                if self.conforming_shapes.contains(&shape_ref.shape_iri) {
                    NodeValidationResult::Conforms
                } else {
                    NodeValidationResult::Violation {
                        shape_iri: shape_ref.shape_iri.clone(),
                        message: format!(
                            "Node <{}> does not conform to shape <{}>",
                            node_iri, shape_ref.shape_iri
                        ),
                    }
                }
            })
            .collect()
    }

    /// Returns `true` if the focus node conforms to **all** shapes referenced
    /// by the constraint.
    pub fn all_conform(&self, node_iri: &str, constraint: &NodeConstraint) -> bool {
        self.validate(node_iri, constraint)
            .iter()
            .all(|r| r.is_conforms())
    }

    /// Returns `true` if the given shape IRI is registered as conforming.
    pub fn is_registered(&self, shape_iri: &str) -> bool {
        self.conforming_shapes.contains(shape_iri)
    }

    /// Returns the total number of registered conforming shapes.
    pub fn registered_count(&self) -> usize {
        self.conforming_shapes.len()
    }
}

impl Default for NodeConstraintValidator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn validator_with_shapes(shapes: &[&str]) -> NodeConstraintValidator {
        shapes.iter().fold(NodeConstraintValidator::new(), |v, s| {
            v.register_conforming(s)
        })
    }

    // --- ShapeRef tests ---

    #[test]
    fn test_shape_ref_new() {
        let r = ShapeRef::new("http://example.org/MyShape");
        assert_eq!(r.shape_iri, "http://example.org/MyShape");
    }

    #[test]
    fn test_shape_ref_equality() {
        let r1 = ShapeRef::new(":Shape1");
        let r2 = ShapeRef::new(":Shape1");
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_shape_ref_inequality() {
        let r1 = ShapeRef::new(":ShapeA");
        let r2 = ShapeRef::new(":ShapeB");
        assert_ne!(r1, r2);
    }

    // --- NodeConstraint tests ---

    #[test]
    fn test_node_constraint_new_empty() {
        let nc = NodeConstraint::new();
        assert_eq!(nc.shape_count(), 0);
        assert!(nc.is_empty());
    }

    #[test]
    fn test_node_constraint_add_shape_single() {
        let nc = NodeConstraint::new().add_shape(":MyShape");
        assert_eq!(nc.shape_count(), 1);
        assert!(nc.contains(":MyShape"));
    }

    #[test]
    fn test_node_constraint_add_shape_multiple() {
        let nc = NodeConstraint::new()
            .add_shape(":ShapeA")
            .add_shape(":ShapeB")
            .add_shape(":ShapeC");
        assert_eq!(nc.shape_count(), 3);
    }

    #[test]
    fn test_node_constraint_contains_true() {
        let nc = NodeConstraint::new().add_shape(":Present");
        assert!(nc.contains(":Present"));
    }

    #[test]
    fn test_node_constraint_contains_false() {
        let nc = NodeConstraint::new().add_shape(":Present");
        assert!(!nc.contains(":Absent"));
    }

    #[test]
    fn test_node_constraint_is_empty_true() {
        let nc = NodeConstraint::new();
        assert!(nc.is_empty());
    }

    #[test]
    fn test_node_constraint_is_empty_false() {
        let nc = NodeConstraint::new().add_shape(":S");
        assert!(!nc.is_empty());
    }

    #[test]
    fn test_node_constraint_iter() {
        let nc = NodeConstraint::new().add_shape(":A").add_shape(":B");
        let iris: Vec<&str> = nc.iter().map(|r| r.shape_iri.as_str()).collect();
        assert!(iris.contains(&":A"));
        assert!(iris.contains(&":B"));
    }

    #[test]
    fn test_node_constraint_default() {
        let nc = NodeConstraint::default();
        assert!(nc.is_empty());
    }

    #[test]
    fn test_node_constraint_shape_refs_order_preserved() {
        let nc = NodeConstraint::new()
            .add_shape(":First")
            .add_shape(":Second");
        assert_eq!(nc.shape_refs[0].shape_iri, ":First");
        assert_eq!(nc.shape_refs[1].shape_iri, ":Second");
    }

    // --- NodeValidationResult tests ---

    #[test]
    fn test_validation_result_conforms_is_conforms() {
        assert!(NodeValidationResult::Conforms.is_conforms());
    }

    #[test]
    fn test_validation_result_conforms_is_not_violation() {
        assert!(!NodeValidationResult::Conforms.is_violation());
    }

    #[test]
    fn test_validation_result_violation_is_violation() {
        let v = NodeValidationResult::Violation {
            shape_iri: ":S".to_string(),
            message: "fail".to_string(),
        };
        assert!(v.is_violation());
        assert!(!v.is_conforms());
    }

    #[test]
    fn test_validation_result_violation_shape_some() {
        let v = NodeValidationResult::Violation {
            shape_iri: ":MyShape".to_string(),
            message: "msg".to_string(),
        };
        assert_eq!(v.violation_shape(), Some(":MyShape"));
    }

    #[test]
    fn test_validation_result_conforms_violation_shape_none() {
        assert_eq!(NodeValidationResult::Conforms.violation_shape(), None);
    }

    #[test]
    fn test_validation_result_equality() {
        let a = NodeValidationResult::Conforms;
        let b = NodeValidationResult::Conforms;
        assert_eq!(a, b);
    }

    // --- NodeConstraintValidator tests ---

    #[test]
    fn test_validator_new_empty() {
        let v = NodeConstraintValidator::new();
        assert_eq!(v.registered_count(), 0);
    }

    #[test]
    fn test_validator_register_conforming() {
        let v = NodeConstraintValidator::new().register_conforming(":S");
        assert!(v.is_registered(":S"));
        assert_eq!(v.registered_count(), 1);
    }

    #[test]
    fn test_validator_register_multiple_shapes() {
        let v = validator_with_shapes(&[":A", ":B", ":C"]);
        assert_eq!(v.registered_count(), 3);
        assert!(v.is_registered(":A"));
        assert!(v.is_registered(":B"));
        assert!(v.is_registered(":C"));
    }

    #[test]
    fn test_validate_empty_constraint_no_results() {
        let v = validator_with_shapes(&[":S"]);
        let nc = NodeConstraint::new();
        let results = v.validate(":node", &nc);
        assert!(results.is_empty());
    }

    #[test]
    fn test_validate_single_conforming_shape() {
        let v = validator_with_shapes(&[":PersonShape"]);
        let nc = NodeConstraint::new().add_shape(":PersonShape");
        let results = v.validate(":alice", &nc);
        assert_eq!(results.len(), 1);
        assert!(results[0].is_conforms());
    }

    #[test]
    fn test_validate_single_non_conforming_shape() {
        let v = NodeConstraintValidator::new();
        let nc = NodeConstraint::new().add_shape(":UnknownShape");
        let results = v.validate(":bob", &nc);
        assert_eq!(results.len(), 1);
        assert!(results[0].is_violation());
        assert_eq!(results[0].violation_shape(), Some(":UnknownShape"));
    }

    #[test]
    fn test_validate_mixed_conforming_and_violating() {
        let v = validator_with_shapes(&[":GoodShape"]);
        let nc = NodeConstraint::new()
            .add_shape(":GoodShape")
            .add_shape(":BadShape");
        let results = v.validate(":node", &nc);
        assert_eq!(results.len(), 2);
        assert!(results[0].is_conforms());
        assert!(results[1].is_violation());
    }

    #[test]
    fn test_all_conform_empty_constraint_true() {
        let v = NodeConstraintValidator::new();
        let nc = NodeConstraint::new();
        assert!(v.all_conform(":any", &nc));
    }

    #[test]
    fn test_all_conform_all_registered() {
        let v = validator_with_shapes(&[":S1", ":S2"]);
        let nc = NodeConstraint::new().add_shape(":S1").add_shape(":S2");
        assert!(v.all_conform(":node", &nc));
    }

    #[test]
    fn test_all_conform_one_unregistered() {
        let v = validator_with_shapes(&[":S1"]);
        let nc = NodeConstraint::new().add_shape(":S1").add_shape(":S2");
        assert!(!v.all_conform(":node", &nc));
    }

    #[test]
    fn test_all_conform_none_registered() {
        let v = NodeConstraintValidator::new();
        let nc = NodeConstraint::new().add_shape(":S1").add_shape(":S2");
        assert!(!v.all_conform(":node", &nc));
    }

    #[test]
    fn test_validate_violation_message_contains_node_and_shape() {
        let v = NodeConstraintValidator::new();
        let nc = NodeConstraint::new().add_shape(":TargetShape");
        let results = v.validate(":focusNode", &nc);
        if let NodeValidationResult::Violation { message, .. } = &results[0] {
            assert!(message.contains(":focusNode"));
            assert!(message.contains(":TargetShape"));
        } else {
            panic!("Expected violation");
        }
    }

    #[test]
    fn test_validator_default_is_empty() {
        let v = NodeConstraintValidator::default();
        assert_eq!(v.registered_count(), 0);
    }

    #[test]
    fn test_validate_three_shapes_all_conforming() {
        let v = validator_with_shapes(&[":S1", ":S2", ":S3"]);
        let nc = NodeConstraint::new()
            .add_shape(":S1")
            .add_shape(":S2")
            .add_shape(":S3");
        let results = v.validate(":n", &nc);
        assert_eq!(results.len(), 3);
        assert!(results.iter().all(|r| r.is_conforms()));
    }

    #[test]
    fn test_validate_three_shapes_all_violating() {
        let v = NodeConstraintValidator::new();
        let nc = NodeConstraint::new()
            .add_shape(":A")
            .add_shape(":B")
            .add_shape(":C");
        let results = v.validate(":n", &nc);
        assert!(results.iter().all(|r| r.is_violation()));
    }

    #[test]
    fn test_is_registered_false() {
        let v = NodeConstraintValidator::new();
        assert!(!v.is_registered(":NotThere"));
    }

    #[test]
    fn test_shape_ref_clone() {
        let r = ShapeRef::new(":CloneMe");
        let r2 = r.clone();
        assert_eq!(r, r2);
    }

    #[test]
    fn test_node_constraint_clone() {
        let nc = NodeConstraint::new().add_shape(":S");
        let nc2 = nc.clone();
        assert_eq!(nc.shape_count(), nc2.shape_count());
    }

    #[test]
    fn test_validate_result_count_matches_shape_refs() {
        let v = validator_with_shapes(&[":A"]);
        let nc = NodeConstraint::new()
            .add_shape(":A")
            .add_shape(":B")
            .add_shape(":C")
            .add_shape(":D");
        let results = v.validate(":x", &nc);
        assert_eq!(results.len(), 4);
    }

    #[test]
    fn test_all_conform_single_shape_registered() {
        let v = validator_with_shapes(&[":OnlyShape"]);
        let nc = NodeConstraint::new().add_shape(":OnlyShape");
        assert!(v.all_conform(":node", &nc));
    }

    #[test]
    fn test_all_conform_single_shape_not_registered() {
        let v = NodeConstraintValidator::new();
        let nc = NodeConstraint::new().add_shape(":Missing");
        assert!(!v.all_conform(":node", &nc));
    }

    #[test]
    fn test_register_same_shape_twice_no_duplicate() {
        let v = NodeConstraintValidator::new()
            .register_conforming(":S")
            .register_conforming(":S");
        // HashSet deduplicates
        assert_eq!(v.registered_count(), 1);
    }

    #[test]
    fn test_validate_different_node_iris_same_result() {
        let v = validator_with_shapes(&[":PersonShape"]);
        let nc = NodeConstraint::new().add_shape(":PersonShape");
        let r1 = v.validate(":alice", &nc);
        let r2 = v.validate(":bob", &nc);
        assert_eq!(r1.len(), r2.len());
        assert!(r1[0].is_conforms());
        assert!(r2[0].is_conforms());
    }

    #[test]
    fn test_node_constraint_contains_after_multiple_adds() {
        let nc = NodeConstraint::new()
            .add_shape(":X")
            .add_shape(":Y")
            .add_shape(":Z");
        assert!(nc.contains(":X"));
        assert!(nc.contains(":Y"));
        assert!(nc.contains(":Z"));
        assert!(!nc.contains(":W"));
    }

    #[test]
    fn test_shape_ref_hash_uniqueness() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(ShapeRef::new(":A"));
        set.insert(ShapeRef::new(":B"));
        set.insert(ShapeRef::new(":A")); // duplicate
        assert_eq!(set.len(), 2);
    }
}
