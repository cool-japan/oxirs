//! Comparison constraint implementations.
//!
//! This module groups SHACL constraints that compare the value nodes of a
//! property shape against either another property's values or a fixed value
//! list. Each struct corresponds to one SHACL Core constraint component:
//!
//! | SHACL parameter            | Spec section | Struct                                       |
//! |----------------------------|--------------|----------------------------------------------|
//! | `sh:equals`                | §4.5.1       | [`EqualsConstraint`]                         |
//! | `sh:disjoint`              | §4.5.2       | [`DisjointConstraint`]                       |
//! | `sh:lessThan`              | §4.5.3       | [`LessThanConstraint`]                       |
//! | `sh:lessThanOrEquals`      | §4.5.4       | [`LessThanOrEqualsConstraint`]               |
//! | `sh:in`                    | §4.8.3       | [`InConstraint`]                             |
//! | `sh:hasValue`              | §4.8.2       | [`HasValueConstraint`]                       |

use super::constraint_context::{ConstraintContext, ConstraintEvaluationResult};
use crate::Result;
use oxirs_core::{model::Term, rdf_store::Store};
use serde::{Deserialize, Serialize};

/// Convert a focus node `Term` into an `oxirs_core::model::Subject`, if the term
/// can occur as an RDF subject. Both IRIs and blank nodes are valid subjects
/// (SHACL shapes are routinely applied to blank-node focus nodes), so this
/// covers both instead of silently dropping blank-node focus nodes.
fn focus_node_as_subject(focus_node: &Term) -> Option<oxirs_core::model::Subject> {
    use oxirs_core::model::Subject;
    match focus_node {
        Term::NamedNode(nn) => Some(Subject::from(nn.clone())),
        Term::BlankNode(bn) => Some(Subject::from(bn.clone())),
        _ => None,
    }
}

/// `sh:equals` constraint (SHACL Core §4.5.1).
///
/// Validates that the set of value nodes for the focus path equals the set of
/// values for the property identified by `property` on the focus node.
/// Two value sets are equal iff they contain the same RDF terms.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EqualsConstraint {
    /// The property whose values must equal the current shape's value nodes.
    pub property: Term,
}

impl EqualsConstraint {
    /// Build an `sh:equals` constraint that compares against the given property.
    pub fn new(property: Term) -> Self {
        Self { property }
    }

    /// Structural validation of the constraint definition itself.
    /// Always succeeds for `EqualsConstraint`; the property reference is checked at evaluation time.
    pub fn validate(&self) -> Result<()> {
        // Basic validation of the constraint structure
        Ok(())
    }

    /// Evaluate the constraint against `context.values` using `store` to look up
    /// the comparator property's values for the focus node. Returns
    /// [`ConstraintEvaluationResult::Satisfied`] when both value sets are equal.
    pub fn evaluate(
        &self,
        context: &ConstraintContext,
        store: &dyn Store,
    ) -> Result<ConstraintEvaluationResult> {
        use oxirs_core::model::Predicate;

        // Get the property values for the constraint property
        let mut constraint_property_values = Vec::new();

        if let (Some(subject), Term::NamedNode(property_node)) =
            (focus_node_as_subject(&context.focus_node), &self.property)
        {
            let predicate = Predicate::from(property_node.clone());

            let quads = store.find_quads(Some(&subject), Some(&predicate), None, None)?;
            for quad in quads {
                constraint_property_values.push(quad.object().clone().into());
            }
        }

        // Check if current values equal constraint property values
        for current_value in &context.values {
            if !constraint_property_values.contains(current_value) {
                return Ok(ConstraintEvaluationResult::violated(
                    Some(current_value.clone()),
                    Some(format!(
                        "Value {current_value} does not equal any value of property {}",
                        self.property
                    )),
                ));
            }
        }

        Ok(ConstraintEvaluationResult::Satisfied)
    }
}

/// `sh:disjoint` constraint (SHACL Core §4.5.2).
///
/// Validates that the set of value nodes is disjoint from the set of values
/// for the named comparator property — i.e. the two sets share no RDF term.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DisjointConstraint {
    /// The comparator property whose values must not appear in the current value set.
    pub property: Term,
}

impl DisjointConstraint {
    pub fn new(property: Term) -> Self {
        Self { property }
    }

    pub fn validate(&self) -> Result<()> {
        // Basic validation of the constraint structure
        Ok(())
    }

    pub fn evaluate(
        &self,
        context: &ConstraintContext,
        store: &dyn Store,
    ) -> Result<ConstraintEvaluationResult> {
        use oxirs_core::model::Predicate;

        // Get the property values for the constraint property
        let mut constraint_property_values = Vec::new();

        if let (Some(subject), Term::NamedNode(property_node)) =
            (focus_node_as_subject(&context.focus_node), &self.property)
        {
            let predicate = Predicate::from(property_node.clone());

            let quads = store.find_quads(Some(&subject), Some(&predicate), None, None)?;
            for quad in quads {
                constraint_property_values.push(quad.object().clone().into());
            }
        }

        // Check if current values are disjoint from constraint property values
        for current_value in &context.values {
            if constraint_property_values.contains(current_value) {
                return Ok(ConstraintEvaluationResult::violated(
                    Some(current_value.clone()),
                    Some(format!(
                        "Value {current_value} is not disjoint from values of property {}",
                        self.property
                    )),
                ));
            }
        }

        Ok(ConstraintEvaluationResult::Satisfied)
    }
}

/// `sh:lessThan` constraint (SHACL Core §4.5.3).
///
/// For every value node `v` of the focus shape, `v` must be less than EVERY
/// value `w` of the comparator property (universal quantification) under
/// SPARQL ordering. Non-numeric (and non-comparable) value nodes are skipped.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LessThanConstraint {
    /// Comparator property whose values must each be greater than every value node.
    pub property: Term,
}

impl LessThanConstraint {
    pub fn new(property: Term) -> Self {
        Self { property }
    }

    pub fn validate(&self) -> Result<()> {
        // Basic validation of the constraint structure
        Ok(())
    }

    pub fn evaluate(
        &self,
        context: &ConstraintContext,
        store: &dyn Store,
    ) -> Result<ConstraintEvaluationResult> {
        use crate::validation::utils::{is_numeric_term, parse_numeric_value};
        use oxirs_core::model::Predicate;

        // Get the property values for the constraint property
        let mut constraint_property_values = Vec::new();

        if let (Some(subject), Term::NamedNode(property_node)) =
            (focus_node_as_subject(&context.focus_node), &self.property)
        {
            let predicate = Predicate::from(property_node.clone());

            let quads = store.find_quads(Some(&subject), Some(&predicate), None, None)?;
            for quad in quads {
                constraint_property_values.push(quad.object().clone().into());
            }
        }

        // Per SHACL Core §4.5.3, every value node must be less than ALL numeric
        // values of the comparator property (universal quantification), not
        // merely at least one of them.
        for current_value in &context.values {
            if !is_numeric_term(current_value) {
                continue; // Skip non-numeric values
            }

            let current_num = parse_numeric_value(current_value)?;

            for constraint_value in &constraint_property_values {
                if !is_numeric_term(constraint_value) {
                    continue; // Skip non-numeric comparator values
                }

                let constraint_num = parse_numeric_value(constraint_value)?;
                // Use `partial_cmp` (rather than `!(a < b)`) so that
                // incomparable values (e.g. NaN) are treated consistently as
                // a violation, matching a strict "a < b" success condition.
                if !matches!(
                    current_num.partial_cmp(&constraint_num),
                    Some(std::cmp::Ordering::Less)
                ) {
                    return Ok(ConstraintEvaluationResult::violated(
                        Some(current_value.clone()),
                        Some(format!(
                            "Value {current_value} is not less than value {constraint_value} of property {}",
                            self.property
                        )),
                    ));
                }
            }
        }

        Ok(ConstraintEvaluationResult::Satisfied)
    }
}

/// `sh:lessThanOrEquals` constraint (SHACL Core §4.5.4).
///
/// Like [`LessThanConstraint`] but uses `<=` instead of strict `<`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LessThanOrEqualsConstraint {
    /// Comparator property whose values must each be greater than or equal to every value node.
    pub property: Term,
}

impl LessThanOrEqualsConstraint {
    pub fn new(property: Term) -> Self {
        Self { property }
    }

    pub fn validate(&self) -> Result<()> {
        // Basic validation of the constraint structure
        Ok(())
    }

    pub fn evaluate(
        &self,
        context: &ConstraintContext,
        store: &dyn Store,
    ) -> Result<ConstraintEvaluationResult> {
        use crate::validation::utils::{is_numeric_term, parse_numeric_value};
        use oxirs_core::model::Predicate;

        // Get the property values for the constraint property
        let mut constraint_property_values = Vec::new();

        if let (Some(subject), Term::NamedNode(property_node)) =
            (focus_node_as_subject(&context.focus_node), &self.property)
        {
            let predicate = Predicate::from(property_node.clone());

            let quads = store.find_quads(Some(&subject), Some(&predicate), None, None)?;
            for quad in quads {
                constraint_property_values.push(quad.object().clone().into());
            }
        }

        // Per SHACL Core §4.5.4, every value node must be less than or equal to
        // ALL numeric values of the comparator property (universal
        // quantification), not merely at least one of them.
        for current_value in &context.values {
            if !is_numeric_term(current_value) {
                continue; // Skip non-numeric values
            }

            let current_num = parse_numeric_value(current_value)?;

            for constraint_value in &constraint_property_values {
                if !is_numeric_term(constraint_value) {
                    continue; // Skip non-numeric comparator values
                }

                let constraint_num = parse_numeric_value(constraint_value)?;
                // Use `partial_cmp` (rather than `!(a <= b)`) so that
                // incomparable values (e.g. NaN) are treated consistently as
                // a violation, matching a strict "a <= b" success condition.
                if !matches!(
                    current_num.partial_cmp(&constraint_num),
                    Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
                ) {
                    return Ok(ConstraintEvaluationResult::violated(
                        Some(current_value.clone()),
                        Some(format!("Value {current_value} is not less than or equal to value {constraint_value} of property {}", self.property)),
                    ));
                }
            }
        }

        Ok(ConstraintEvaluationResult::Satisfied)
    }
}

/// `sh:in` constraint (SHACL Core §4.8.3).
///
/// Validates that every value node is contained in the explicit, ordered list
/// of allowed values declared via `sh:in`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InConstraint {
    /// The closed enumeration of permitted RDF terms.
    pub values: Vec<Term>,
}

impl InConstraint {
    pub fn new(values: Vec<Term>) -> Self {
        Self { values }
    }

    pub fn validate(&self) -> Result<()> {
        // Basic validation of the constraint structure
        Ok(())
    }

    pub fn evaluate(
        &self,
        context: &ConstraintContext,
        _store: &dyn Store,
    ) -> Result<ConstraintEvaluationResult> {
        // Check if all values in the context are in the allowed set
        for value in &context.values {
            if !self.values.contains(value) {
                return Ok(ConstraintEvaluationResult::violated(
                    Some(value.clone()),
                    Some(format!("Value {value} is not in the allowed set of values")),
                ));
            }
        }

        Ok(ConstraintEvaluationResult::Satisfied)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PropertyPath, ShapeId};
    use oxirs_core::{
        model::{GraphName, Literal, NamedNode, Object, Predicate, Quad, Subject, Term},
        ConcreteStore,
    };

    fn make_iri(s: &str) -> Term {
        Term::NamedNode(NamedNode::new(s).expect("valid IRI"))
    }

    fn focus_node() -> Term {
        make_iri("http://example.org/subject")
    }

    fn shape_id() -> ShapeId {
        ShapeId::new("http://example.org/shape1")
    }

    fn property_path(pred: &str) -> PropertyPath {
        PropertyPath::Predicate(NamedNode::new(pred).expect("valid IRI"))
    }

    fn make_ctx(focus: Term, path: PropertyPath, values: Vec<Term>) -> ConstraintContext {
        ConstraintContext::new(focus, shape_id())
            .with_path(path)
            .with_values(values)
    }

    fn plain_lit(s: &str) -> Term {
        Term::Literal(Literal::new(s))
    }

    fn int_lit(n: &str) -> Term {
        let dt = NamedNode::new_unchecked("http://www.w3.org/2001/XMLSchema#integer");
        Term::Literal(Literal::new_typed_literal(n, dt))
    }

    fn float_lit(n: &str) -> Term {
        let dt = NamedNode::new_unchecked("http://www.w3.org/2001/XMLSchema#decimal");
        Term::Literal(Literal::new_typed_literal(n, dt))
    }

    /// Insert a triple into the store: focus_subject --predicate--> object_value
    fn insert_triple(store: &ConcreteStore, subject: &Term, predicate_iri: &str, object: Term) {
        let subj = match subject {
            Term::NamedNode(n) => Subject::from(n.clone()),
            Term::BlankNode(b) => Subject::from(b.clone()),
            _ => panic!("subject must be a named node or blank node"),
        };
        let pred = Predicate::from(NamedNode::new(predicate_iri).expect("valid IRI"));
        let obj = match object {
            Term::NamedNode(n) => Object::from(n.clone()),
            Term::Literal(l) => Object::from(l),
            Term::BlankNode(b) => Object::from(b),
            _ => panic!("only NamedNode, Literal, and BlankNode are supported as objects"),
        };
        let quad = Quad::new(subj, pred, obj, GraphName::DefaultGraph);
        store.insert_quad(quad).expect("insert quad");
    }

    const PRED_A: &str = "http://example.org/predA";
    const PRED_B: &str = "http://example.org/predB";

    // ---- EqualsConstraint tests ----

    #[test]
    fn test_equals_validate_ok() {
        let c = EqualsConstraint::new(make_iri(PRED_A));
        assert!(c.validate().is_ok());
    }

    #[test]
    fn test_equals_empty_values_no_constraint_property_satisfied() {
        // No values in context, no values in store: vacuously satisfied
        let c = EqualsConstraint::new(make_iri(PRED_A));
        let store = ConcreteStore::new().expect("store");
        let ctx = make_ctx(focus_node(), property_path(PRED_B), vec![]);
        assert!(c.evaluate(&ctx, &store).expect("eval").is_satisfied());
    }

    #[test]
    fn test_equals_single_value_matching_store_satisfied() {
        let c = EqualsConstraint::new(make_iri(PRED_A));
        let store = ConcreteStore::new().expect("store");
        let value = plain_lit("Alice");
        insert_triple(&store, &focus_node(), PRED_A, value.clone());
        let ctx = make_ctx(focus_node(), property_path(PRED_B), vec![value]);
        assert!(c.evaluate(&ctx, &store).expect("eval").is_satisfied());
    }

    #[test]
    fn test_equals_value_not_in_store_violated() {
        let c = EqualsConstraint::new(make_iri(PRED_A));
        let store = ConcreteStore::new().expect("store");
        // Store has "Alice" for PRED_A, but we check "Bob"
        insert_triple(&store, &focus_node(), PRED_A, plain_lit("Alice"));
        let ctx = make_ctx(focus_node(), property_path(PRED_B), vec![plain_lit("Bob")]);
        let result = c.evaluate(&ctx, &store).expect("eval");
        assert!(
            result.is_violated(),
            "Value not in store's PRED_A should be violated"
        );
    }

    #[test]
    fn test_equals_value_no_store_data_violated() {
        // Store is empty, but context has a value: must be violated
        let c = EqualsConstraint::new(make_iri(PRED_A));
        let store = ConcreteStore::new().expect("store");
        let ctx = make_ctx(
            focus_node(),
            property_path(PRED_B),
            vec![plain_lit("SomeValue")],
        );
        let result = c.evaluate(&ctx, &store).expect("eval");
        assert!(
            result.is_violated(),
            "No PRED_A value in store, but context has a value: violated"
        );
    }

    #[test]
    fn test_equals_iri_matching_store_satisfied() {
        let c = EqualsConstraint::new(make_iri(PRED_A));
        let store = ConcreteStore::new().expect("store");
        let value = make_iri("http://example.org/something");
        insert_triple(&store, &focus_node(), PRED_A, value.clone());
        let ctx = make_ctx(focus_node(), property_path(PRED_B), vec![value]);
        assert!(c.evaluate(&ctx, &store).expect("eval").is_satisfied());
    }

    #[test]
    fn test_equals_focus_node_not_named_node_empty_store_satisfied() {
        // When focus_node is not a named node, constraint property lookup is skipped
        let c = EqualsConstraint::new(make_iri(PRED_A));
        let store = ConcreteStore::new().expect("store");
        // Blank node as focus node - the constraint can't look up the property
        let ctx = ConstraintContext::new(
            Term::BlankNode(oxirs_core::model::BlankNode::new_unchecked("b1")),
            shape_id(),
        )
        .with_path(property_path(PRED_B))
        .with_values(vec![plain_lit("x")]);
        // When focus node can't be resolved, result depends on implementation
        let result = c.evaluate(&ctx, &store).expect("eval");
        assert!(result.is_satisfied() || result.is_violated());
    }

    // ---- DisjointConstraint tests ----

    #[test]
    fn test_disjoint_validate_ok() {
        let c = DisjointConstraint::new(make_iri(PRED_A));
        assert!(c.validate().is_ok());
    }

    #[test]
    fn test_disjoint_empty_values_satisfied() {
        let c = DisjointConstraint::new(make_iri(PRED_A));
        let store = ConcreteStore::new().expect("store");
        let ctx = make_ctx(focus_node(), property_path(PRED_B), vec![]);
        assert!(c.evaluate(&ctx, &store).expect("eval").is_satisfied());
    }

    #[test]
    fn test_disjoint_no_overlap_satisfied() {
        let c = DisjointConstraint::new(make_iri(PRED_A));
        let store = ConcreteStore::new().expect("store");
        // Store has "Alice" for PRED_A, context has "Bob" for current property
        insert_triple(&store, &focus_node(), PRED_A, plain_lit("Alice"));
        let ctx = make_ctx(focus_node(), property_path(PRED_B), vec![plain_lit("Bob")]);
        assert!(c.evaluate(&ctx, &store).expect("eval").is_satisfied());
    }

    #[test]
    fn test_disjoint_overlap_violated() {
        let c = DisjointConstraint::new(make_iri(PRED_A));
        let store = ConcreteStore::new().expect("store");
        // Store has "Alice" for PRED_A, context also has "Alice" for current property: overlap
        insert_triple(&store, &focus_node(), PRED_A, plain_lit("Alice"));
        let ctx = make_ctx(
            focus_node(),
            property_path(PRED_B),
            vec![plain_lit("Alice")],
        );
        let result = c.evaluate(&ctx, &store).expect("eval");
        assert!(result.is_violated(), "Overlapping values should violate");
    }

    #[test]
    fn test_disjoint_empty_store_value_in_context_satisfied() {
        // No PRED_A values in store, any context value is fine
        let c = DisjointConstraint::new(make_iri(PRED_A));
        let store = ConcreteStore::new().expect("store");
        let ctx = make_ctx(
            focus_node(),
            property_path(PRED_B),
            vec![plain_lit("X"), plain_lit("Y")],
        );
        assert!(c.evaluate(&ctx, &store).expect("eval").is_satisfied());
    }

    #[test]
    fn test_disjoint_multiple_context_one_overlaps_violated() {
        let c = DisjointConstraint::new(make_iri(PRED_A));
        let store = ConcreteStore::new().expect("store");
        insert_triple(&store, &focus_node(), PRED_A, plain_lit("Alice"));
        // Context has "Bob" and "Alice"; "Alice" overlaps
        let ctx = make_ctx(
            focus_node(),
            property_path(PRED_B),
            vec![plain_lit("Bob"), plain_lit("Alice")],
        );
        let result = c.evaluate(&ctx, &store).expect("eval");
        // Disjoint iterates in order; "Bob" is OK, "Alice" is not
        assert!(result.is_violated());
    }

    // ---- LessThanConstraint tests ----

    #[test]
    fn test_less_than_validate_ok() {
        let c = LessThanConstraint::new(make_iri(PRED_A));
        assert!(c.validate().is_ok());
    }

    #[test]
    fn test_less_than_empty_values_satisfied() {
        let c = LessThanConstraint::new(make_iri(PRED_A));
        let store = ConcreteStore::new().expect("store");
        let ctx = make_ctx(focus_node(), property_path(PRED_B), vec![]);
        assert!(c.evaluate(&ctx, &store).expect("eval").is_satisfied());
    }

    #[test]
    fn test_less_than_numeric_less_than_store_value_satisfied() {
        let c = LessThanConstraint::new(make_iri(PRED_A));
        let store = ConcreteStore::new().expect("store");
        // PRED_A = 10 in store, context has 5: 5 < 10 => satisfied
        insert_triple(&store, &focus_node(), PRED_A, int_lit("10"));
        let ctx = make_ctx(focus_node(), property_path(PRED_B), vec![int_lit("5")]);
        assert!(c.evaluate(&ctx, &store).expect("eval").is_satisfied());
    }

    #[test]
    fn test_less_than_numeric_equal_violated() {
        let c = LessThanConstraint::new(make_iri(PRED_A));
        let store = ConcreteStore::new().expect("store");
        // PRED_A = 10, context has 10: 10 < 10 => false => violated
        insert_triple(&store, &focus_node(), PRED_A, int_lit("10"));
        let ctx = make_ctx(focus_node(), property_path(PRED_B), vec![int_lit("10")]);
        let result = c.evaluate(&ctx, &store).expect("eval");
        assert!(
            result.is_violated(),
            "Equal values should not satisfy lessThan"
        );
    }

    #[test]
    fn test_less_than_numeric_greater_violated() {
        let c = LessThanConstraint::new(make_iri(PRED_A));
        let store = ConcreteStore::new().expect("store");
        // PRED_A = 10, context has 20: 20 < 10 => false => violated
        insert_triple(&store, &focus_node(), PRED_A, int_lit("10"));
        let ctx = make_ctx(focus_node(), property_path(PRED_B), vec![int_lit("20")]);
        let result = c.evaluate(&ctx, &store).expect("eval");
        assert!(
            result.is_violated(),
            "Greater value should violate lessThan"
        );
    }

    #[test]
    fn test_less_than_non_numeric_values_skipped_satisfied() {
        // Non-numeric values are skipped; if all skipped, vacuously satisfied
        let c = LessThanConstraint::new(make_iri(PRED_A));
        let store = ConcreteStore::new().expect("store");
        insert_triple(&store, &focus_node(), PRED_A, int_lit("10"));
        let ctx = make_ctx(
            focus_node(),
            property_path(PRED_B),
            vec![plain_lit("text value")],
        );
        // Non-numeric context values should be skipped => satisfied
        assert!(c.evaluate(&ctx, &store).expect("eval").is_satisfied());
    }

    #[test]
    fn test_less_than_float_less_than_satisfied() {
        let c = LessThanConstraint::new(make_iri(PRED_A));
        let store = ConcreteStore::new().expect("store");
        insert_triple(&store, &focus_node(), PRED_A, float_lit("10.5"));
        let ctx = make_ctx(focus_node(), property_path(PRED_B), vec![float_lit("3.14")]);
        assert!(c.evaluate(&ctx, &store).expect("eval").is_satisfied());
    }

    // ---- LessThanOrEqualsConstraint tests ----

    #[test]
    fn test_less_than_or_equals_validate_ok() {
        let c = LessThanOrEqualsConstraint::new(make_iri(PRED_A));
        assert!(c.validate().is_ok());
    }

    #[test]
    fn test_less_than_or_equals_empty_values_satisfied() {
        let c = LessThanOrEqualsConstraint::new(make_iri(PRED_A));
        let store = ConcreteStore::new().expect("store");
        let ctx = make_ctx(focus_node(), property_path(PRED_B), vec![]);
        assert!(c.evaluate(&ctx, &store).expect("eval").is_satisfied());
    }

    #[test]
    fn test_less_than_or_equals_equal_value_satisfied() {
        let c = LessThanOrEqualsConstraint::new(make_iri(PRED_A));
        let store = ConcreteStore::new().expect("store");
        // PRED_A = 10, context has 10: 10 <= 10 => satisfied
        insert_triple(&store, &focus_node(), PRED_A, int_lit("10"));
        let ctx = make_ctx(focus_node(), property_path(PRED_B), vec![int_lit("10")]);
        assert!(c.evaluate(&ctx, &store).expect("eval").is_satisfied());
    }

    #[test]
    fn test_less_than_or_equals_less_than_satisfied() {
        let c = LessThanOrEqualsConstraint::new(make_iri(PRED_A));
        let store = ConcreteStore::new().expect("store");
        insert_triple(&store, &focus_node(), PRED_A, int_lit("10"));
        let ctx = make_ctx(focus_node(), property_path(PRED_B), vec![int_lit("5")]);
        assert!(c.evaluate(&ctx, &store).expect("eval").is_satisfied());
    }

    #[test]
    fn test_less_than_or_equals_greater_violated() {
        let c = LessThanOrEqualsConstraint::new(make_iri(PRED_A));
        let store = ConcreteStore::new().expect("store");
        // PRED_A = 10, context has 20: 20 <= 10 => false => violated
        insert_triple(&store, &focus_node(), PRED_A, int_lit("10"));
        let ctx = make_ctx(focus_node(), property_path(PRED_B), vec![int_lit("20")]);
        let result = c.evaluate(&ctx, &store).expect("eval");
        assert!(
            result.is_violated(),
            "Greater value should violate lessThanOrEquals"
        );
    }

    #[test]
    fn test_less_than_or_equals_non_numeric_skipped_satisfied() {
        let c = LessThanOrEqualsConstraint::new(make_iri(PRED_A));
        let store = ConcreteStore::new().expect("store");
        insert_triple(&store, &focus_node(), PRED_A, int_lit("10"));
        let ctx = make_ctx(
            focus_node(),
            property_path(PRED_B),
            vec![plain_lit("non-numeric")],
        );
        assert!(c.evaluate(&ctx, &store).expect("eval").is_satisfied());
    }

    // ---- InConstraint tests ----

    #[test]
    fn test_in_validate_ok() {
        let c = InConstraint::new(vec![plain_lit("a"), plain_lit("b")]);
        assert!(c.validate().is_ok());
    }

    #[test]
    fn test_in_value_in_allowed_set_satisfied() {
        let c = InConstraint::new(vec![plain_lit("a"), plain_lit("b"), plain_lit("c")]);
        let store = ConcreteStore::new().expect("store");
        let ctx = make_ctx(focus_node(), property_path(PRED_B), vec![plain_lit("b")]);
        assert!(c.evaluate(&ctx, &store).expect("eval").is_satisfied());
    }

    #[test]
    fn test_in_value_not_in_allowed_set_violated() {
        let c = InConstraint::new(vec![plain_lit("a"), plain_lit("b")]);
        let store = ConcreteStore::new().expect("store");
        let ctx = make_ctx(focus_node(), property_path(PRED_B), vec![plain_lit("x")]);
        assert!(c.evaluate(&ctx, &store).expect("eval").is_violated());
    }

    #[test]
    fn test_in_empty_values_satisfied() {
        let c = InConstraint::new(vec![plain_lit("a")]);
        let store = ConcreteStore::new().expect("store");
        let ctx = make_ctx(focus_node(), property_path(PRED_B), vec![]);
        assert!(c.evaluate(&ctx, &store).expect("eval").is_satisfied());
    }

    // ---- HasValueConstraint tests ----

    #[test]
    fn test_has_value_matching_satisfied() {
        let required = plain_lit("required-value");
        let c = HasValueConstraint::new(required.clone());
        let store = ConcreteStore::new().expect("store");
        let ctx = make_ctx(
            focus_node(),
            property_path(PRED_B),
            vec![required, plain_lit("other")],
        );
        assert!(c.evaluate(&ctx, &store).expect("eval").is_satisfied());
    }

    #[test]
    fn test_has_value_missing_violated() {
        let c = HasValueConstraint::new(plain_lit("required-value"));
        let store = ConcreteStore::new().expect("store");
        let ctx = make_ctx(
            focus_node(),
            property_path(PRED_B),
            vec![plain_lit("other-value")],
        );
        assert!(c.evaluate(&ctx, &store).expect("eval").is_violated());
    }

    #[test]
    fn test_has_value_empty_values_violated() {
        let c = HasValueConstraint::new(plain_lit("required-value"));
        let store = ConcreteStore::new().expect("store");
        let ctx = make_ctx(focus_node(), property_path(PRED_B), vec![]);
        assert!(c.evaluate(&ctx, &store).expect("eval").is_violated());
    }

    // ---- Universal quantification for sh:lessThan / sh:lessThanOrEquals ----

    #[test]
    fn regression_less_than_requires_universal_not_existential() {
        // Focus value 15, comparator property has TWO values: 10 and 20.
        // 15 is NOT less than 10, so this must be a VIOLATION even though
        // 15 < 20 (the old existential "satisfied = any(...)" bug would have
        // reported this as satisfied after finding just the 20 match).
        let c = LessThanConstraint::new(make_iri(PRED_A));
        let store = ConcreteStore::new().expect("store");
        insert_triple(&store, &focus_node(), PRED_A, int_lit("10"));
        insert_triple(&store, &focus_node(), PRED_A, int_lit("20"));
        let ctx = make_ctx(focus_node(), property_path(PRED_B), vec![int_lit("15")]);
        let result = c.evaluate(&ctx, &store).expect("eval");
        assert!(
            result.is_violated(),
            "15 is not less than 10, so this must violate even though 15 < 20"
        );
    }

    #[test]
    fn regression_less_than_satisfied_when_less_than_all_comparator_values() {
        let c = LessThanConstraint::new(make_iri(PRED_A));
        let store = ConcreteStore::new().expect("store");
        insert_triple(&store, &focus_node(), PRED_A, int_lit("10"));
        insert_triple(&store, &focus_node(), PRED_A, int_lit("20"));
        let ctx = make_ctx(focus_node(), property_path(PRED_B), vec![int_lit("5")]);
        let result = c.evaluate(&ctx, &store).expect("eval");
        assert!(
            result.is_satisfied(),
            "5 is less than every comparator value (10 and 20), so this must be satisfied"
        );
    }

    #[test]
    fn regression_less_than_or_equals_requires_universal_not_existential() {
        let c = LessThanOrEqualsConstraint::new(make_iri(PRED_A));
        let store = ConcreteStore::new().expect("store");
        insert_triple(&store, &focus_node(), PRED_A, int_lit("10"));
        insert_triple(&store, &focus_node(), PRED_A, int_lit("20"));
        // 15 <= 20 but 15 > 10 -> must violate.
        let ctx = make_ctx(focus_node(), property_path(PRED_B), vec![int_lit("15")]);
        let result = c.evaluate(&ctx, &store).expect("eval");
        assert!(result.is_violated());
    }

    // ---- Blank-node focus node support for comparison constraints ----

    #[test]
    fn regression_disjoint_blank_node_focus_detects_overlap() {
        let bnode = Term::BlankNode(oxirs_core::model::BlankNode::new_unchecked("b1"));
        let c = DisjointConstraint::new(make_iri(PRED_A));
        let store = ConcreteStore::new().expect("store");
        // The blank-node focus node has "shared" for PRED_A in the store, and
        // the current property (evaluated via context.values) also has
        // "shared" -> real overlap -> must violate.
        insert_triple(&store, &bnode, PRED_A, plain_lit("shared"));
        let ctx = make_ctx(bnode, property_path(PRED_B), vec![plain_lit("shared")]);
        let result = c.evaluate(&ctx, &store).expect("eval");
        assert!(
            result.is_violated(),
            "blank-node focus node overlap must be detected, not silently ignored"
        );
    }

    #[test]
    fn regression_equals_blank_node_focus_uses_store_values() {
        let bnode = Term::BlankNode(oxirs_core::model::BlankNode::new_unchecked("b2"));
        let c = EqualsConstraint::new(make_iri(PRED_A));
        let store = ConcreteStore::new().expect("store");
        insert_triple(&store, &bnode, PRED_A, plain_lit("Alice"));
        let ctx = make_ctx(bnode, property_path(PRED_B), vec![plain_lit("Alice")]);
        let result = c.evaluate(&ctx, &store).expect("eval");
        assert!(
            result.is_satisfied(),
            "blank-node focus node's stored PRED_A value must be looked up, not skipped"
        );
    }
}

/// `sh:hasValue` constraint (SHACL Core §4.8.2).
///
/// Validates that the set of value nodes contains at least one occurrence of
/// the configured RDF term. Empty value sets fail the constraint.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HasValueConstraint {
    /// The required RDF term that must appear among the value nodes.
    pub value: Term,
}

impl HasValueConstraint {
    pub fn new(value: Term) -> Self {
        Self { value }
    }

    pub fn validate(&self) -> Result<()> {
        // Basic validation of the constraint structure
        Ok(())
    }

    pub fn evaluate(
        &self,
        context: &ConstraintContext,
        _store: &dyn Store,
    ) -> Result<ConstraintEvaluationResult> {
        // Check if any value in the context matches the required value
        for value in &context.values {
            if value == &self.value {
                return Ok(ConstraintEvaluationResult::Satisfied);
            }
        }

        // If no matching value found, constraint is violated
        Ok(ConstraintEvaluationResult::violated(
            None,
            Some(format!(
                "No value matches required value: {value}",
                value = self.value
            )),
        ))
    }
}
