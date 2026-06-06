//! SHACL Expression Constraints (sh:ExpressionConstraintComponent)
//!
//! Implements the SHACL Advanced Features expression mechanism, which allows
//! evaluating mathematical and string expressions over RDF node values.
//!
//! Reference: <https://www.w3.org/TR/shacl-af/#ExpressionConstraintComponent>
//!
//! This module is a thin facade. The implementation is split across:
//! - [`crate::constraints::expression_constraint_types`] — the value, path, and
//!   AST types ([`ShaclPath`], [`ShaclValue`], [`ShaclExpression`]).
//! - [`crate::constraints::expression_constraint_evaluator`] — the evaluation
//!   environment ([`ExpressionContext`], [`PathResolver`]) and the
//!   [`ExpressionEvaluator`].
//! - [`crate::constraints::expression_constraint_component`] — the
//!   [`ExpressionConstraintComponent`] and [`ExpressionConstraintResult`].

pub use super::expression_constraint_component::*;
pub use super::expression_constraint_evaluator::*;
pub use super::expression_constraint_types::*;
