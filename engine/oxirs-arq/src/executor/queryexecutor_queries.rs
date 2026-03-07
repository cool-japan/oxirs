//! # QueryExecutor - queries Methods
//!
//! This module contains method implementations for `QueryExecutor`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::Result;
use std::cell::RefCell;
use tracing::debug;

use super::queryexecutor_type::QueryExecutor;

// Thread-local storage for the current dataset pointer during filter evaluation.
// This allows EXISTS/NOT EXISTS subquery evaluation to access the dataset
// without requiring a major refactor of the evaluate_expression signature.
// We store a raw pointer as usize to work around lifetime constraints.
thread_local! {
    // Stores (data_ptr, vtable_ptr) pair for *const dyn Dataset
    pub(crate) static EXISTS_DATASET: RefCell<Option<(usize, usize)>> =
        const { RefCell::new(None) };
}

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
                // Date/time functions
                "now" | "NOW" | "http://www.w3.org/2005/xpath-functions#current-dateTime" => {
                    self.datetime_now_function()
                }
                "year" | "YEAR" | "http://www.w3.org/2005/xpath-functions#year-from-dateTime" => {
                    if args.len() == 1 {
                        let arg = self.evaluate_expression(&args[0], binding)?;
                        self.datetime_component_function(&arg, "year")
                    } else {
                        Err(anyhow::anyhow!("year() requires exactly 1 argument"))
                    }
                }
                "month"
                | "MONTH"
                | "http://www.w3.org/2005/xpath-functions#month-from-dateTime" => {
                    if args.len() == 1 {
                        let arg = self.evaluate_expression(&args[0], binding)?;
                        self.datetime_component_function(&arg, "month")
                    } else {
                        Err(anyhow::anyhow!("month() requires exactly 1 argument"))
                    }
                }
                "day" | "DAY" | "http://www.w3.org/2005/xpath-functions#day-from-dateTime" => {
                    if args.len() == 1 {
                        let arg = self.evaluate_expression(&args[0], binding)?;
                        self.datetime_component_function(&arg, "day")
                    } else {
                        Err(anyhow::anyhow!("day() requires exactly 1 argument"))
                    }
                }
                "hours"
                | "HOURS"
                | "http://www.w3.org/2005/xpath-functions#hours-from-dateTime" => {
                    if args.len() == 1 {
                        let arg = self.evaluate_expression(&args[0], binding)?;
                        self.datetime_component_function(&arg, "hours")
                    } else {
                        Err(anyhow::anyhow!("hours() requires exactly 1 argument"))
                    }
                }
                "minutes"
                | "MINUTES"
                | "http://www.w3.org/2005/xpath-functions#minutes-from-dateTime" => {
                    if args.len() == 1 {
                        let arg = self.evaluate_expression(&args[0], binding)?;
                        self.datetime_component_function(&arg, "minutes")
                    } else {
                        Err(anyhow::anyhow!("minutes() requires exactly 1 argument"))
                    }
                }
                "seconds"
                | "SECONDS"
                | "http://www.w3.org/2005/xpath-functions#seconds-from-dateTime" => {
                    if args.len() == 1 {
                        let arg = self.evaluate_expression(&args[0], binding)?;
                        self.datetime_component_function(&arg, "seconds")
                    } else {
                        Err(anyhow::anyhow!("seconds() requires exactly 1 argument"))
                    }
                }
                "timezone"
                | "TIMEZONE"
                | "http://www.w3.org/2005/xpath-functions#timezone-from-dateTime" => {
                    if args.len() == 1 {
                        let arg = self.evaluate_expression(&args[0], binding)?;
                        self.datetime_component_function(&arg, "timezone")
                    } else {
                        Err(anyhow::anyhow!("timezone() requires exactly 1 argument"))
                    }
                }
                "tz" | "TZ" | "http://www.w3.org/2005/xpath-functions#tz" => {
                    if args.len() == 1 {
                        let arg = self.evaluate_expression(&args[0], binding)?;
                        self.datetime_component_function(&arg, "tz")
                    } else {
                        Err(anyhow::anyhow!("tz() requires exactly 1 argument"))
                    }
                }
                // String functions
                "strlen" | "STRLEN" => {
                    if args.len() == 1 {
                        let arg = self.evaluate_expression(&args[0], binding)?;
                        self.strlen_function(&arg)
                    } else {
                        Err(anyhow::anyhow!("strlen() requires exactly 1 argument"))
                    }
                }
                "ucase" | "UCASE" => {
                    if args.len() == 1 {
                        let arg = self.evaluate_expression(&args[0], binding)?;
                        self.ucase_function(&arg)
                    } else {
                        Err(anyhow::anyhow!("ucase() requires exactly 1 argument"))
                    }
                }
                "lcase" | "LCASE" => {
                    if args.len() == 1 {
                        let arg = self.evaluate_expression(&args[0], binding)?;
                        self.lcase_function(&arg)
                    } else {
                        Err(anyhow::anyhow!("lcase() requires exactly 1 argument"))
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
        // Attempt to use the thread-local dataset if available for proper evaluation
        let result = EXISTS_DATASET.with(|cell| {
            let raw_opt = *cell.borrow();
            let (data_ptr, vtable_ptr) = match raw_opt {
                Some(ptrs) => ptrs,
                None => return None,
            };
            // Safety: the dataset pointer is set by apply_filter_with_dataset before calling
            // evaluate_expression, which calls this function. The dataset outlives
            // the filter evaluation call chain (stored on stack above us).
            let fat_ptr: *const dyn super::dataset::Dataset =
                unsafe { std::mem::transmute((data_ptr, vtable_ptr)) };
            let dataset: &dyn super::dataset::Dataset = unsafe { &*fat_ptr };
            // Execute the subquery against the dataset, substituting current binding
            // into the algebra by first executing, then checking if any result
            // is compatible with current_binding
            let mut joined_algebra = algebra.clone();
            // Substitute known bindings as Values inline
            if !current_binding.is_empty() {
                let vars: Vec<crate::algebra::Variable> = current_binding.keys().cloned().collect();
                let bindings_vec = vec![current_binding.clone()];
                let values_node = crate::algebra::Algebra::Values {
                    variables: vars,
                    bindings: bindings_vec,
                };
                joined_algebra = crate::algebra::Algebra::Join {
                    left: Box::new(values_node),
                    right: Box::new(algebra.clone()),
                };
            }
            match self.execute_serial(&joined_algebra, dataset) {
                Ok(solutions) => Some(Ok(!solutions.is_empty())),
                Err(e) => Some(Err(e)),
            }
        });

        if let Some(r) = result {
            return r;
        }

        // Fallback: syntactic pattern check (when no dataset available)
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
            BinaryOperator::Add => {
                let left_num = self.extract_numeric_value(left)?;
                let right_num = self.extract_numeric_value(right)?;
                let result = left_num + right_num;
                // Use integer type if both values are integers
                let (value_str, datatype) =
                    if result.fract() == 0.0 && left_num.fract() == 0.0 && right_num.fract() == 0.0
                    {
                        (
                            format!("{}", result as i64),
                            "http://www.w3.org/2001/XMLSchema#integer",
                        )
                    } else {
                        (
                            format!("{}", result),
                            "http://www.w3.org/2001/XMLSchema#decimal",
                        )
                    };
                Ok(Term::Literal(crate::algebra::Literal {
                    value: value_str,
                    language: None,
                    datatype: Some(oxirs_core::model::NamedNode::new_unchecked(datatype)),
                }))
            }
            BinaryOperator::Subtract => {
                let left_num = self.extract_numeric_value(left)?;
                let right_num = self.extract_numeric_value(right)?;
                let result = left_num - right_num;
                let (value_str, datatype) =
                    if result.fract() == 0.0 && left_num.fract() == 0.0 && right_num.fract() == 0.0
                    {
                        (
                            format!("{}", result as i64),
                            "http://www.w3.org/2001/XMLSchema#integer",
                        )
                    } else {
                        (
                            format!("{}", result),
                            "http://www.w3.org/2001/XMLSchema#decimal",
                        )
                    };
                Ok(Term::Literal(crate::algebra::Literal {
                    value: value_str,
                    language: None,
                    datatype: Some(oxirs_core::model::NamedNode::new_unchecked(datatype)),
                }))
            }
            BinaryOperator::Multiply => {
                let left_num = self.extract_numeric_value(left)?;
                let right_num = self.extract_numeric_value(right)?;
                let result = left_num * right_num;
                let (value_str, datatype) =
                    if result.fract() == 0.0 && left_num.fract() == 0.0 && right_num.fract() == 0.0
                    {
                        (
                            format!("{}", result as i64),
                            "http://www.w3.org/2001/XMLSchema#integer",
                        )
                    } else {
                        (
                            format!("{}", result),
                            "http://www.w3.org/2001/XMLSchema#decimal",
                        )
                    };
                Ok(Term::Literal(crate::algebra::Literal {
                    value: value_str,
                    language: None,
                    datatype: Some(oxirs_core::model::NamedNode::new_unchecked(datatype)),
                }))
            }
            BinaryOperator::Divide => {
                let left_num = self.extract_numeric_value(left)?;
                let right_num = self.extract_numeric_value(right)?;
                if right_num == 0.0 {
                    return Err(anyhow::anyhow!("Division by zero"));
                }
                let result = left_num / right_num;
                let (value_str, datatype) = if result.fract() == 0.0 {
                    (
                        format!("{}", result as i64),
                        "http://www.w3.org/2001/XMLSchema#integer",
                    )
                } else {
                    (
                        format!("{}", result),
                        "http://www.w3.org/2001/XMLSchema#decimal",
                    )
                };
                Ok(Term::Literal(crate::algebra::Literal {
                    value: value_str,
                    language: None,
                    datatype: Some(oxirs_core::model::NamedNode::new_unchecked(datatype)),
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
                // Use as_str() to get the IRI value WITHOUT angle brackets
                // (iri.to_string() returns "<http://...>" which is SPARQL notation)
                Ok(crate::algebra::Term::Literal(crate::algebra::Literal {
                    value: iri.as_str().to_string(),
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

    // ===== Date/Time Helper Functions =====

    /// Return the current date/time as an xsd:dateTime literal
    pub(super) fn datetime_now_function(&self) -> Result<crate::algebra::Term> {
        use chrono::Utc;
        let now = Utc::now();
        let formatted = now.format("%Y-%m-%dT%H:%M:%SZ").to_string();
        Ok(crate::algebra::Term::Literal(crate::algebra::Literal {
            value: formatted,
            language: None,
            datatype: Some(oxirs_core::model::NamedNode::new_unchecked(
                "http://www.w3.org/2001/XMLSchema#dateTime",
            )),
        }))
    }

    /// Extract a named component from an xsd:dateTime literal.
    ///
    /// `component` is one of: "year", "month", "day", "hours", "minutes",
    /// "seconds", "timezone", "tz".
    pub(super) fn datetime_component_function(
        &self,
        arg: &crate::algebra::Term,
        component: &str,
    ) -> Result<crate::algebra::Term> {
        use chrono::{DateTime, Datelike, Timelike, Utc};

        let raw = match arg {
            crate::algebra::Term::Literal(lit) => lit.value.as_str(),
            _ => {
                return Err(anyhow::anyhow!(
                    "{component}() requires a dateTime literal argument"
                ));
            }
        };

        // Parse the raw dateTime string
        let dt = raw
            .parse::<DateTime<Utc>>()
            .map_err(|e| anyhow::anyhow!("Failed to parse dateTime '{}': {}", raw, e))?;

        match component {
            "year" => {
                let yr = dt.year() as i64;
                Ok(crate::algebra::Term::Literal(crate::algebra::Literal {
                    value: yr.to_string(),
                    language: None,
                    datatype: Some(oxirs_core::model::NamedNode::new_unchecked(
                        "http://www.w3.org/2001/XMLSchema#integer",
                    )),
                }))
            }
            "month" => {
                let mo = dt.month() as i64;
                Ok(crate::algebra::Term::Literal(crate::algebra::Literal {
                    value: mo.to_string(),
                    language: None,
                    datatype: Some(oxirs_core::model::NamedNode::new_unchecked(
                        "http://www.w3.org/2001/XMLSchema#integer",
                    )),
                }))
            }
            "day" => {
                let d = dt.day() as i64;
                Ok(crate::algebra::Term::Literal(crate::algebra::Literal {
                    value: d.to_string(),
                    language: None,
                    datatype: Some(oxirs_core::model::NamedNode::new_unchecked(
                        "http://www.w3.org/2001/XMLSchema#integer",
                    )),
                }))
            }
            "hours" => {
                let h = dt.hour() as i64;
                Ok(crate::algebra::Term::Literal(crate::algebra::Literal {
                    value: h.to_string(),
                    language: None,
                    datatype: Some(oxirs_core::model::NamedNode::new_unchecked(
                        "http://www.w3.org/2001/XMLSchema#integer",
                    )),
                }))
            }
            "minutes" => {
                let m = dt.minute() as i64;
                Ok(crate::algebra::Term::Literal(crate::algebra::Literal {
                    value: m.to_string(),
                    language: None,
                    datatype: Some(oxirs_core::model::NamedNode::new_unchecked(
                        "http://www.w3.org/2001/XMLSchema#integer",
                    )),
                }))
            }
            "seconds" => {
                let s = dt.second() as f64 + dt.nanosecond() as f64 / 1_000_000_000.0;
                Ok(crate::algebra::Term::Literal(crate::algebra::Literal {
                    value: s.to_string(),
                    language: None,
                    datatype: Some(oxirs_core::model::NamedNode::new_unchecked(
                        "http://www.w3.org/2001/XMLSchema#decimal",
                    )),
                }))
            }
            "timezone" => {
                // For UTC datetimes return PT0S (zero duration), for offset return the offset xsd:dayTimeDuration
                let tz_str = "PT0S".to_string();
                Ok(crate::algebra::Term::Literal(crate::algebra::Literal {
                    value: tz_str,
                    language: None,
                    datatype: Some(oxirs_core::model::NamedNode::new_unchecked(
                        "http://www.w3.org/2001/XMLSchema#dayTimeDuration",
                    )),
                }))
            }
            "tz" => {
                // TZ() returns the timezone offset as a plain string (e.g. "Z", "+05:30")
                let tz_str = if raw.ends_with('Z') {
                    "Z".to_string()
                } else if let Some(pos) = raw.rfind(['+', '-']) {
                    if pos > 10 {
                        // It's an offset part like +05:30
                        raw[pos..].to_string()
                    } else {
                        String::new()
                    }
                } else {
                    String::new()
                };
                Ok(crate::algebra::Term::Literal(crate::algebra::Literal {
                    value: tz_str,
                    language: None,
                    datatype: Some(oxirs_core::model::NamedNode::new_unchecked(
                        "http://www.w3.org/2001/XMLSchema#string",
                    )),
                }))
            }
            other => Err(anyhow::anyhow!("Unknown datetime component: {}", other)),
        }
    }

    /// Built-in STRLEN function
    pub(super) fn strlen_function(
        &self,
        arg: &crate::algebra::Term,
    ) -> Result<crate::algebra::Term> {
        let s = match arg {
            crate::algebra::Term::Literal(lit) => lit.value.as_str(),
            _ => {
                return Err(anyhow::anyhow!(
                    "strlen() requires a string or literal argument"
                ));
            }
        };
        let len = s.chars().count() as i64;
        Ok(crate::algebra::Term::Literal(crate::algebra::Literal {
            value: len.to_string(),
            language: None,
            datatype: Some(oxirs_core::model::NamedNode::new_unchecked(
                "http://www.w3.org/2001/XMLSchema#integer",
            )),
        }))
    }

    /// Built-in UCASE function
    pub(super) fn ucase_function(
        &self,
        arg: &crate::algebra::Term,
    ) -> Result<crate::algebra::Term> {
        match arg {
            crate::algebra::Term::Literal(lit) => {
                Ok(crate::algebra::Term::Literal(crate::algebra::Literal {
                    value: lit.value.to_uppercase(),
                    language: lit.language.clone(),
                    datatype: lit.datatype.clone(),
                }))
            }
            _ => Err(anyhow::anyhow!(
                "ucase() requires a string literal argument"
            )),
        }
    }

    /// Built-in LCASE function
    pub(super) fn lcase_function(
        &self,
        arg: &crate::algebra::Term,
    ) -> Result<crate::algebra::Term> {
        match arg {
            crate::algebra::Term::Literal(lit) => {
                Ok(crate::algebra::Term::Literal(crate::algebra::Literal {
                    value: lit.value.to_lowercase(),
                    language: lit.language.clone(),
                    datatype: lit.datatype.clone(),
                }))
            }
            _ => Err(anyhow::anyhow!(
                "lcase() requires a string literal argument"
            )),
        }
    }
}
