//! SHACL cardinality constraints for validating the number of values
//!
//! This module implements cardinality constraints that validate the number of values
//! associated with a property on focus nodes:
//!
//! - [`MinCountConstraint`] - Validates minimum number of values (`sh:minCount`)
//! - [`MaxCountConstraint`] - Validates maximum number of values (`sh:maxCount`)
//!
//! # Usage
//!
//! ```rust
//! use oxirs_shacl::constraints::cardinality_constraints::*;
//!
//! // Create a constraint requiring at least 1 value
//! let min_constraint = MinCountConstraint {
//!     min_count: 1,
//! };
//!
//! // Create a constraint allowing at most 3 values
//! let max_constraint = MaxCountConstraint {
//!     max_count: 3,
//! };
//!
//! // Common pattern: exactly one value (min=1, max=1)
//! let exactly_one_min = MinCountConstraint { min_count: 1 };
//! let exactly_one_max = MaxCountConstraint { max_count: 1 };
//! ```
//!
//! # SHACL Specification
//!
//! These constraints implement the cardinality constraint components from the
//! [SHACL specification](https://www.w3.org/TR/shacl/#core-components-count):
//!
//! - `sh:minCount` - Specifies the minimum number of values in the set of value nodes
//! - `sh:maxCount` - Specifies the maximum number of values in the set of value nodes

use serde::{Deserialize, Serialize};

use oxirs_core::Store;

use super::{
    ConstraintContext, ConstraintEvaluationResult, ConstraintEvaluator, ConstraintValidator,
};
use crate::Result;

/// SHACL `sh:minCount` constraint that validates the minimum number of values.
///
/// This constraint ensures that there are at least the specified number of values
/// for the property being validated. It's commonly used to enforce required properties
/// and minimum cardinality relationships.
///
/// # SHACL Specification
///
/// From [SHACL Core Components - MinCount Constraint Component](https://www.w3.org/TR/shacl/#MinCountConstraintComponent):
/// "Specifies the minimum number of values in the set of value nodes."
///
/// # Example
///
/// ```rust
/// use oxirs_shacl::constraints::cardinality_constraints::MinCountConstraint;
///
/// // Require at least one value (mandatory property)
/// let required_constraint = MinCountConstraint {
///     min_count: 1,
/// };
///
/// // Require at least 2 values (e.g., for a "hasAuthor" property that needs multiple authors)
/// let multiple_values_constraint = MinCountConstraint {
///     min_count: 2,
/// };
/// ```
///
/// # Validation Behavior
///
/// - **Passes**: When the number of values is greater than or equal to `min_count`
/// - **Fails**: When the number of values is less than `min_count`
/// - **Edge Case**: A `min_count` of 0 will always pass (equivalent to no constraint)
///
/// # Common Use Cases
///
/// - **Required Properties**: Set `min_count: 1` to make a property mandatory
/// - **Minimum Relationships**: Ensure entities have at least N relationships
/// - **Data Completeness**: Validate that multi-valued properties have sufficient data
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MinCountConstraint {
    /// The minimum number of values required
    pub min_count: u32,
}

impl ConstraintValidator for MinCountConstraint {
    fn validate(&self) -> Result<()> {
        Ok(())
    }
}

impl ConstraintEvaluator for MinCountConstraint {
    fn evaluate(
        &self,
        _store: &dyn Store,
        context: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult> {
        let value_count = context.values.len() as u32;
        if value_count < self.min_count {
            return Ok(ConstraintEvaluationResult::violated(
                None,
                Some(format!(
                    "Expected at least {} values, but found {}",
                    self.min_count, value_count
                )),
            ));
        }
        Ok(ConstraintEvaluationResult::satisfied())
    }
}

/// SHACL `sh:maxCount` constraint that validates the maximum number of values.
///
/// This constraint ensures that there are at most the specified number of values
/// for the property being validated. It's commonly used to enforce single-valued
/// properties and maximum cardinality relationships.
///
/// # SHACL Specification
///
/// From [SHACL Core Components - MaxCount Constraint Component](https://www.w3.org/TR/shacl/#MaxCountConstraintComponent):
/// "Specifies the maximum number of values in the set of value nodes."
///
/// # Example
///
/// ```rust
/// use oxirs_shacl::constraints::cardinality_constraints::MaxCountConstraint;
///
/// // Allow at most one value (single-valued property)
/// let single_value_constraint = MaxCountConstraint {
///     max_count: 1,
/// };
///
/// // Allow at most 5 values (e.g., for a "hasTag" property with limited tags)
/// let limited_values_constraint = MaxCountConstraint {
///     max_count: 5,
/// };
/// ```
///
/// # Validation Behavior
///
/// - **Passes**: When the number of values is less than or equal to `max_count`
/// - **Fails**: When the number of values is greater than `max_count`
/// - **Edge Case**: A `max_count` of 0 requires no values (empty property)
///
/// # Common Use Cases
///
/// - **Single-Valued Properties**: Set `max_count: 1` to ensure functional properties
/// - **Limited Relationships**: Restrict the number of connections an entity can have
/// - **Resource Constraints**: Limit multi-valued properties to prevent unbounded growth
/// - **Business Rules**: Enforce domain-specific cardinality limits
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaxCountConstraint {
    /// The maximum number of values allowed
    pub max_count: u32,
}

impl ConstraintValidator for MaxCountConstraint {
    fn validate(&self) -> Result<()> {
        Ok(())
    }
}

impl ConstraintEvaluator for MaxCountConstraint {
    fn evaluate(
        &self,
        _store: &dyn Store,
        context: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult> {
        let value_count = context.values.len() as u32;
        if value_count > self.max_count {
            return Ok(ConstraintEvaluationResult::violated(
                None,
                Some(format!(
                    "Expected at most {} values, but found {}",
                    self.max_count, value_count
                )),
            ));
        }
        Ok(ConstraintEvaluationResult::satisfied())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PropertyPath, ShapeId};
    use oxirs_core::{
        model::{Literal, NamedNode, Term},
        ConcreteStore,
    };

    fn make_focus_node() -> Term {
        Term::NamedNode(NamedNode::new("http://example.org/node1").expect("valid IRI"))
    }

    fn make_shape_id() -> ShapeId {
        ShapeId::new("http://example.org/shape1")
    }

    fn make_property_path() -> PropertyPath {
        PropertyPath::Predicate(NamedNode::new("http://example.org/name").expect("valid IRI"))
    }

    fn string_term(s: &str) -> Term {
        Term::Literal(Literal::new(s))
    }

    // ---- MinCountConstraint tests ----

    #[test]
    fn test_min_count_satisfied_exact() {
        let constraint = MinCountConstraint { min_count: 2 };
        let store = ConcreteStore::new().expect("store creation");
        let context = ConstraintContext::new(make_focus_node(), make_shape_id())
            .with_path(make_property_path())
            .with_values(vec![string_term("Alice"), string_term("Bob")]);

        let result = constraint.evaluate(&store, &context).expect("evaluation");
        assert!(
            result.is_satisfied(),
            "Expected satisfied for exact min count"
        );
    }

    #[test]
    fn test_min_count_satisfied_more_than_required() {
        let constraint = MinCountConstraint { min_count: 1 };
        let store = ConcreteStore::new().expect("store creation");
        let context = ConstraintContext::new(make_focus_node(), make_shape_id())
            .with_path(make_property_path())
            .with_values(vec![string_term("a"), string_term("b"), string_term("c")]);

        let result = constraint.evaluate(&store, &context).expect("evaluation");
        assert!(
            result.is_satisfied(),
            "Expected satisfied for more than min count"
        );
    }

    #[test]
    fn test_min_count_violated_zero_values() {
        let constraint = MinCountConstraint { min_count: 1 };
        let store = ConcreteStore::new().expect("store creation");
        let context = ConstraintContext::new(make_focus_node(), make_shape_id())
            .with_path(make_property_path())
            .with_values(vec![]);

        let result = constraint.evaluate(&store, &context).expect("evaluation");
        assert!(
            result.is_violated(),
            "Expected violation for zero values when min_count=1"
        );
        assert!(result.message().is_some(), "Expected violation message");
    }

    #[test]
    fn test_min_count_violated_insufficient_values() {
        let constraint = MinCountConstraint { min_count: 3 };
        let store = ConcreteStore::new().expect("store creation");
        let context = ConstraintContext::new(make_focus_node(), make_shape_id())
            .with_path(make_property_path())
            .with_values(vec![string_term("a"), string_term("b")]);

        let result = constraint.evaluate(&store, &context).expect("evaluation");
        assert!(
            result.is_violated(),
            "Expected violation for insufficient values"
        );
    }

    #[test]
    fn test_min_count_zero_always_satisfied() {
        let constraint = MinCountConstraint { min_count: 0 };
        let store = ConcreteStore::new().expect("store creation");
        let context = ConstraintContext::new(make_focus_node(), make_shape_id())
            .with_path(make_property_path())
            .with_values(vec![]);

        let result = constraint.evaluate(&store, &context).expect("evaluation");
        assert!(
            result.is_satisfied(),
            "min_count=0 should always be satisfied"
        );
    }

    #[test]
    fn test_min_count_validate_ok() {
        let constraint = MinCountConstraint { min_count: 1 };
        assert!(
            constraint.validate().is_ok(),
            "MinCountConstraint should validate OK"
        );
    }

    // ---- MaxCountConstraint tests ----

    #[test]
    fn test_max_count_satisfied_zero_values() {
        let constraint = MaxCountConstraint { max_count: 3 };
        let store = ConcreteStore::new().expect("store creation");
        let context = ConstraintContext::new(make_focus_node(), make_shape_id())
            .with_path(make_property_path())
            .with_values(vec![]);

        let result = constraint.evaluate(&store, &context).expect("evaluation");
        assert!(result.is_satisfied(), "Expected satisfied for no values");
    }

    #[test]
    fn test_max_count_satisfied_exact() {
        let constraint = MaxCountConstraint { max_count: 2 };
        let store = ConcreteStore::new().expect("store creation");
        let context = ConstraintContext::new(make_focus_node(), make_shape_id())
            .with_path(make_property_path())
            .with_values(vec![string_term("a"), string_term("b")]);

        let result = constraint.evaluate(&store, &context).expect("evaluation");
        assert!(
            result.is_satisfied(),
            "Expected satisfied for exact max count"
        );
    }

    #[test]
    fn test_max_count_violated() {
        let constraint = MaxCountConstraint { max_count: 1 };
        let store = ConcreteStore::new().expect("store creation");
        let context = ConstraintContext::new(make_focus_node(), make_shape_id())
            .with_path(make_property_path())
            .with_values(vec![string_term("a"), string_term("b")]);

        let result = constraint.evaluate(&store, &context).expect("evaluation");
        assert!(
            result.is_violated(),
            "Expected violation for too many values"
        );
        assert!(result.message().is_some(), "Expected violation message");
    }

    #[test]
    fn test_max_count_zero_requires_no_values() {
        let constraint = MaxCountConstraint { max_count: 0 };
        let store = ConcreteStore::new().expect("store creation");

        let empty_context = ConstraintContext::new(make_focus_node(), make_shape_id())
            .with_path(make_property_path())
            .with_values(vec![]);
        let result = constraint
            .evaluate(&store, &empty_context)
            .expect("evaluation");
        assert!(
            result.is_satisfied(),
            "max_count=0 with no values should be satisfied"
        );

        let non_empty_context = ConstraintContext::new(make_focus_node(), make_shape_id())
            .with_path(make_property_path())
            .with_values(vec![string_term("value")]);
        let result2 = constraint
            .evaluate(&store, &non_empty_context)
            .expect("evaluation");
        assert!(
            result2.is_violated(),
            "max_count=0 with one value should be violated"
        );
    }

    #[test]
    fn test_max_count_validate_ok() {
        let constraint = MaxCountConstraint { max_count: 5 };
        assert!(
            constraint.validate().is_ok(),
            "MaxCountConstraint should validate OK"
        );
    }

    // ---- Combined min/max (exactly-one pattern) ----

    #[test]
    fn test_exactly_one_value_satisfied() {
        let min_constraint = MinCountConstraint { min_count: 1 };
        let max_constraint = MaxCountConstraint { max_count: 1 };
        let store = ConcreteStore::new().expect("store creation");
        let context = ConstraintContext::new(make_focus_node(), make_shape_id())
            .with_path(make_property_path())
            .with_values(vec![string_term("single value")]);

        let min_result = min_constraint
            .evaluate(&store, &context)
            .expect("evaluation");
        let max_result = max_constraint
            .evaluate(&store, &context)
            .expect("evaluation");

        assert!(
            min_result.is_satisfied(),
            "min=1 should be satisfied for exactly one value"
        );
        assert!(
            max_result.is_satisfied(),
            "max=1 should be satisfied for exactly one value"
        );
    }

    #[test]
    fn test_exactly_one_value_violated_by_zero() {
        let min_constraint = MinCountConstraint { min_count: 1 };
        let store = ConcreteStore::new().expect("store creation");
        let context = ConstraintContext::new(make_focus_node(), make_shape_id())
            .with_path(make_property_path())
            .with_values(vec![]);

        let result = min_constraint
            .evaluate(&store, &context)
            .expect("evaluation");
        assert!(result.is_violated(), "Exactly-one violated by zero values");
    }

    #[test]
    fn test_exactly_one_value_violated_by_two() {
        let max_constraint = MaxCountConstraint { max_count: 1 };
        let store = ConcreteStore::new().expect("store creation");
        let context = ConstraintContext::new(make_focus_node(), make_shape_id())
            .with_path(make_property_path())
            .with_values(vec![string_term("v1"), string_term("v2")]);

        let result = max_constraint
            .evaluate(&store, &context)
            .expect("evaluation");
        assert!(result.is_violated(), "Exactly-one violated by two values");
    }
}
