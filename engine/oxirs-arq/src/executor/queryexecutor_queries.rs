//! # QueryExecutor - queries Methods
//!
//! This module contains method implementations for `QueryExecutor`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::Result;
use tracing::debug;

use super::queryexecutor_type::QueryExecutor;

impl QueryExecutor {
    /// Evaluate a SPARQL expression against a binding
    pub(super) fn evaluate_expression(
        &self,
        expr: &crate::algebra::Expression,
        binding: &crate::algebra::Binding,
    ) -> Result<crate::algebra::Term> {
        use crate::algebra::Expression;
        match expr {
            Expression::Variable(var) => binding
                .get(var)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("Unbound variable: {}", var)),
            Expression::Literal(lit) => Ok(crate::algebra::Term::Literal(lit.clone())),
            Expression::Iri(iri) => Ok(crate::algebra::Term::Iri(iri.clone())),
            Expression::Binary { op, left, right } => {
                let left_val = self.evaluate_expression(left, binding)?;
                let right_val = self.evaluate_expression(right, binding)?;
                self.evaluate_binary_operation(op, &left_val, &right_val)
            }
            Expression::Unary { op, operand } => {
                let val = self.evaluate_expression(operand, binding)?;
                self.evaluate_unary_operation(op, &val)
            }
            Expression::Bound(var) => {
                let is_bound = binding.contains_key(var);
                Ok(crate::algebra::Term::Literal(crate::algebra::Literal {
                    value: is_bound.to_string(),
                    language: None,
                    datatype: Some(oxirs_core::model::NamedNode::new_unchecked(
                        "http://www.w3.org/2001/XMLSchema#boolean",
                    )),
                }))
            }
            Expression::Conditional {
                condition,
                then_expr,
                else_expr,
            } => {
                let condition_result = self.evaluate_expression(condition, binding)?;
                if let crate::algebra::Term::Literal(lit) = condition_result {
                    if self.is_truthy(&lit) {
                        self.evaluate_expression(then_expr, binding)
                    } else {
                        self.evaluate_expression(else_expr, binding)
                    }
                } else {
                    self.evaluate_expression(then_expr, binding)
                }
            }
            Expression::Function { name, args } => match name.as_str() {
                "str" => {
                    if args.len() == 1 {
                        let arg = self.evaluate_expression(&args[0], binding)?;
                        self.str_function(&arg)
                    } else {
                        Err(anyhow::anyhow!(
                            "str() function requires exactly 1 argument"
                        ))
                    }
                }
                "lang" => {
                    if args.len() == 1 {
                        let arg = self.evaluate_expression(&args[0], binding)?;
                        self.lang_function(&arg)
                    } else {
                        Err(anyhow::anyhow!(
                            "lang() function requires exactly 1 argument"
                        ))
                    }
                }
                "datatype" => {
                    if args.len() == 1 {
                        let arg = self.evaluate_expression(&args[0], binding)?;
                        self.datatype_function(&arg)
                    } else {
                        Err(anyhow::anyhow!(
                            "datatype() function requires exactly 1 argument"
                        ))
                    }
                }
                _ => Err(anyhow::anyhow!("Unknown function: {}", name)),
            },
            Expression::Exists(algebra) => match self.evaluate_exists_subquery(algebra, binding) {
                Ok(has_solutions) => Ok(crate::algebra::Term::Literal(crate::algebra::Literal {
                    value: has_solutions.to_string(),
                    language: None,
                    datatype: Some(oxirs_core::model::NamedNode::new_unchecked(
                        "http://www.w3.org/2001/XMLSchema#boolean",
                    )),
                })),
                Err(e) => {
                    debug!("EXISTS subquery evaluation failed: {}", e);
                    Ok(crate::algebra::Term::Literal(crate::algebra::Literal {
                        value: "false".to_string(),
                        language: None,
                        datatype: Some(oxirs_core::model::NamedNode::new_unchecked(
                            "http://www.w3.org/2001/XMLSchema#boolean",
                        )),
                    }))
                }
            },
            Expression::NotExists(algebra) => {
                match self.evaluate_exists_subquery(algebra, binding) {
                    Ok(has_solutions) => {
                        Ok(crate::algebra::Term::Literal(crate::algebra::Literal {
                            value: (!has_solutions).to_string(),
                            language: None,
                            datatype: Some(oxirs_core::model::NamedNode::new_unchecked(
                                "http://www.w3.org/2001/XMLSchema#boolean",
                            )),
                        }))
                    }
                    Err(e) => {
                        debug!("NOT EXISTS subquery evaluation failed: {}", e);
                        Ok(crate::algebra::Term::Literal(crate::algebra::Literal {
                            value: "true".to_string(),
                            language: None,
                            datatype: Some(oxirs_core::model::NamedNode::new_unchecked(
                                "http://www.w3.org/2001/XMLSchema#boolean",
                            )),
                        }))
                    }
                }
            }
        }
    }
    /// Evaluate EXISTS/NOT EXISTS subquery
    pub(super) fn evaluate_exists_subquery(
        &self,
        algebra: &crate::algebra::Algebra,
        current_binding: &crate::algebra::Binding,
    ) -> Result<bool> {
        use crate::algebra::Algebra;
        match algebra {
            Algebra::Bgp(patterns) => {
                for pattern in patterns {
                    let test_binding = current_binding.clone();
                    if self.pattern_might_match(pattern, &test_binding) {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
            Algebra::Filter {
                pattern,
                condition: _,
            } => self.evaluate_exists_subquery(pattern, current_binding),
            Algebra::Union { left, right } => {
                let left_result = self.evaluate_exists_subquery(left, current_binding)?;
                if left_result {
                    return Ok(true);
                }
                self.evaluate_exists_subquery(right, current_binding)
            }
            Algebra::Join { left, right } => {
                let left_result = self.evaluate_exists_subquery(left, current_binding)?;
                if !left_result {
                    return Ok(false);
                }
                self.evaluate_exists_subquery(right, current_binding)
            }
            _ => Ok(false),
        }
    }
    /// Evaluate binary operations
    pub(super) fn evaluate_binary_operation(
        &self,
        op: &crate::algebra::BinaryOperator,
        left: &crate::algebra::Term,
        right: &crate::algebra::Term,
    ) -> Result<crate::algebra::Term> {
        use crate::algebra::{BinaryOperator, Term};
        match op {
            BinaryOperator::Equal => {
                let result = self.terms_equal(left, right);
                Ok(Term::Literal(crate::algebra::Literal {
                    value: result.to_string(),
                    language: None,
                    datatype: Some(oxirs_core::model::NamedNode::new_unchecked(
                        "http://www.w3.org/2001/XMLSchema#boolean",
                    )),
                }))
            }
            BinaryOperator::NotEqual => {
                let result = !self.terms_equal(left, right);
                Ok(Term::Literal(crate::algebra::Literal {
                    value: result.to_string(),
                    language: None,
                    datatype: Some(oxirs_core::model::NamedNode::new_unchecked(
                        "http://www.w3.org/2001/XMLSchema#boolean",
                    )),
                }))
            }
            BinaryOperator::Less => self.numeric_comparison(left, right, |a, b| a < b),
            BinaryOperator::LessEqual => self.numeric_comparison(left, right, |a, b| a <= b),
            BinaryOperator::Greater => self.numeric_comparison(left, right, |a, b| a > b),
            BinaryOperator::GreaterEqual => self.numeric_comparison(left, right, |a, b| a >= b),
            BinaryOperator::And => {
                let left_truth = self.is_term_truthy(left)?;
                let right_truth = self.is_term_truthy(right)?;
                Ok(Term::Literal(crate::algebra::Literal {
                    value: (left_truth && right_truth).to_string(),
                    language: None,
                    datatype: Some(oxirs_core::model::NamedNode::new_unchecked(
                        "http://www.w3.org/2001/XMLSchema#boolean",
                    )),
                }))
            }
            BinaryOperator::Or => {
                let left_truth = self.is_term_truthy(left)?;
                let right_truth = self.is_term_truthy(right)?;
                Ok(Term::Literal(crate::algebra::Literal {
                    value: (left_truth || right_truth).to_string(),
                    language: None,
                    datatype: Some(oxirs_core::model::NamedNode::new_unchecked(
                        "http://www.w3.org/2001/XMLSchema#boolean",
                    )),
                }))
            }
            BinaryOperator::SameTerm => {
                let result = left == right;
                Ok(Term::Literal(crate::algebra::Literal {
                    value: result.to_string(),
                    language: None,
                    datatype: Some(oxirs_core::model::NamedNode::new_unchecked(
                        "http://www.w3.org/2001/XMLSchema#boolean",
                    )),
                }))
            }
            _ => Err(anyhow::anyhow!(
                "Binary operator {:?} not yet implemented",
                op
            )),
        }
    }
    /// Evaluate unary operations
    pub(super) fn evaluate_unary_operation(
        &self,
        op: &crate::algebra::UnaryOperator,
        operand: &crate::algebra::Term,
    ) -> Result<crate::algebra::Term> {
        use crate::algebra::{Term, UnaryOperator};
        match op {
            UnaryOperator::Not => {
                let truth = self.is_term_truthy(operand)?;
                Ok(Term::Literal(crate::algebra::Literal {
                    value: (!truth).to_string(),
                    language: None,
                    datatype: Some(oxirs_core::model::NamedNode::new_unchecked(
                        "http://www.w3.org/2001/XMLSchema#boolean",
                    )),
                }))
            }
            UnaryOperator::IsIri => {
                let result = matches!(operand, Term::Iri(_));
                Ok(Term::Literal(crate::algebra::Literal {
                    value: result.to_string(),
                    language: None,
                    datatype: Some(oxirs_core::model::NamedNode::new_unchecked(
                        "http://www.w3.org/2001/XMLSchema#boolean",
                    )),
                }))
            }
            UnaryOperator::IsBlank => {
                let result = matches!(operand, Term::BlankNode(_));
                Ok(Term::Literal(crate::algebra::Literal {
                    value: result.to_string(),
                    language: None,
                    datatype: Some(oxirs_core::model::NamedNode::new_unchecked(
                        "http://www.w3.org/2001/XMLSchema#boolean",
                    )),
                }))
            }
            UnaryOperator::IsLiteral => {
                let result = matches!(operand, Term::Literal(_));
                Ok(Term::Literal(crate::algebra::Literal {
                    value: result.to_string(),
                    language: None,
                    datatype: Some(oxirs_core::model::NamedNode::new_unchecked(
                        "http://www.w3.org/2001/XMLSchema#boolean",
                    )),
                }))
            }
            UnaryOperator::IsNumeric => {
                let result = self.is_numeric_literal(operand);
                Ok(Term::Literal(crate::algebra::Literal {
                    value: result.to_string(),
                    language: None,
                    datatype: Some(oxirs_core::model::NamedNode::new_unchecked(
                        "http://www.w3.org/2001/XMLSchema#boolean",
                    )),
                }))
            }
            _ => Err(anyhow::anyhow!(
                "Unary operator {:?} not yet implemented",
                op
            )),
        }
    }
    /// Check if a literal is truthy
    pub(super) fn is_truthy(&self, literal: &crate::algebra::Literal) -> bool {
        if let Some(ref datatype) = literal.datatype {
            match datatype.as_str() {
                "http://www.w3.org/2001/XMLSchema#boolean" => {
                    literal.value == "true" || literal.value == "1"
                }
                "http://www.w3.org/2001/XMLSchema#integer"
                | "http://www.w3.org/2001/XMLSchema#decimal"
                | "http://www.w3.org/2001/XMLSchema#double"
                | "http://www.w3.org/2001/XMLSchema#float" => literal
                    .value
                    .parse::<f64>()
                    .map(|n| n != 0.0)
                    .unwrap_or(false),
                "http://www.w3.org/2001/XMLSchema#string" => !literal.value.is_empty(),
                _ => !literal.value.is_empty(),
            }
        } else {
            !literal.value.is_empty()
        }
    }
    /// Built-in STR function
    pub(super) fn str_function(&self, arg: &crate::algebra::Term) -> Result<crate::algebra::Term> {
        match arg {
            crate::algebra::Term::Literal(lit) => {
                Ok(crate::algebra::Term::Literal(crate::algebra::Literal {
                    value: lit.value.clone(),
                    language: None,
                    datatype: Some(oxirs_core::model::NamedNode::new_unchecked(
                        "http://www.w3.org/2001/XMLSchema#string",
                    )),
                }))
            }
            crate::algebra::Term::Iri(iri) => {
                Ok(crate::algebra::Term::Literal(crate::algebra::Literal {
                    value: iri.to_string(),
                    language: None,
                    datatype: Some(oxirs_core::model::NamedNode::new_unchecked(
                        "http://www.w3.org/2001/XMLSchema#string",
                    )),
                }))
            }
            _ => Err(anyhow::anyhow!(
                "STR function not applicable to this term type"
            )),
        }
    }
    /// Built-in DATATYPE function
    pub(super) fn datatype_function(
        &self,
        arg: &crate::algebra::Term,
    ) -> Result<crate::algebra::Term> {
        match arg {
            crate::algebra::Term::Literal(lit) => {
                let datatype = lit
                    .datatype
                    .as_ref()
                    .unwrap_or(&oxirs_core::model::NamedNode::new_unchecked(
                        "http://www.w3.org/2001/XMLSchema#string",
                    ))
                    .clone();
                Ok(crate::algebra::Term::Iri(datatype))
            }
            _ => Err(anyhow::anyhow!(
                "DATATYPE function only applicable to literals"
            )),
        }
    }
}
