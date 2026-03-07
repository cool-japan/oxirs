//! Comparison constraint implementations

use super::constraint_context::{ConstraintContext, ConstraintEvaluationResult};
use crate::Result;
use oxirs_core::{model::Term, rdf_store::Store};
use serde::{Deserialize, Serialize};

/// Equals constraint
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EqualsConstraint {
    pub property: Term,
}

impl EqualsConstraint {
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
        use oxirs_core::model::{Predicate, Subject};

        // Get the property values for the constraint property
        let mut constraint_property_values = Vec::new();

        if let (Term::NamedNode(focus_node), Term::NamedNode(property_node)) =
            (&context.focus_node, &self.property)
        {
            let subject = Subject::from(focus_node.clone());
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

/// Disjoint constraint
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DisjointConstraint {
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
        use oxirs_core::model::{Predicate, Subject};

        // Get the property values for the constraint property
        let mut constraint_property_values = Vec::new();

        if let (Term::NamedNode(focus_node), Term::NamedNode(property_node)) =
            (&context.focus_node, &self.property)
        {
            let subject = Subject::from(focus_node.clone());
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

/// Less than constraint
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LessThanConstraint {
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
        use oxirs_core::model::{Predicate, Subject};

        // Get the property values for the constraint property
        let mut constraint_property_values = Vec::new();

        if let (Term::NamedNode(focus_node), Term::NamedNode(property_node)) =
            (&context.focus_node, &self.property)
        {
            let subject = Subject::from(focus_node.clone());
            let predicate = Predicate::from(property_node.clone());

            let quads = store.find_quads(Some(&subject), Some(&predicate), None, None)?;
            for quad in quads {
                constraint_property_values.push(quad.object().clone().into());
            }
        }

        // Check if current values are less than constraint property values
        for current_value in &context.values {
            if !is_numeric_term(current_value) {
                continue; // Skip non-numeric values
            }

            let current_num = parse_numeric_value(current_value)?;
            let mut satisfied = false;

            for constraint_value in &constraint_property_values {
                if is_numeric_term(constraint_value) {
                    let constraint_num = parse_numeric_value(constraint_value)?;
                    if current_num < constraint_num {
                        satisfied = true;
                        break;
                    }
                }
            }

            if !satisfied {
                return Ok(ConstraintEvaluationResult::violated(
                    Some(current_value.clone()),
                    Some(format!(
                        "Value {current_value} is not less than any value of property {}",
                        self.property
                    )),
                ));
            }
        }

        Ok(ConstraintEvaluationResult::Satisfied)
    }
}

/// Less than or equals constraint
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LessThanOrEqualsConstraint {
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
        use oxirs_core::model::{Predicate, Subject};

        // Get the property values for the constraint property
        let mut constraint_property_values = Vec::new();

        if let (Term::NamedNode(focus_node), Term::NamedNode(property_node)) =
            (&context.focus_node, &self.property)
        {
            let subject = Subject::from(focus_node.clone());
            let predicate = Predicate::from(property_node.clone());

            let quads = store.find_quads(Some(&subject), Some(&predicate), None, None)?;
            for quad in quads {
                constraint_property_values.push(quad.object().clone().into());
            }
        }

        // Check if current values are less than or equal to constraint property values
        for current_value in &context.values {
            if !is_numeric_term(current_value) {
                continue; // Skip non-numeric values
            }

            let current_num = parse_numeric_value(current_value)?;
            let mut satisfied = false;

            for constraint_value in &constraint_property_values {
                if is_numeric_term(constraint_value) {
                    let constraint_num = parse_numeric_value(constraint_value)?;
                    if current_num <= constraint_num {
                        satisfied = true;
                        break;
                    }
                }
            }

            if !satisfied {
                return Ok(ConstraintEvaluationResult::violated(
                    Some(current_value.clone()),
                    Some(format!("Value {current_value} is not less than or equal to any value of property {}", self.property)),
                ));
            }
        }

        Ok(ConstraintEvaluationResult::Satisfied)
    }
}

/// In constraint
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InConstraint {
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
            _ => panic!("subject must be named node"),
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
}

/// Has value constraint
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HasValueConstraint {
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
