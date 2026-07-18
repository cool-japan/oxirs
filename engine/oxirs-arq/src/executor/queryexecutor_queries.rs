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
            Expression::Binary { op, left, right }
                if matches!(
                    op,
                    crate::algebra::BinaryOperator::In | crate::algebra::BinaryOperator::NotIn
                ) =>
            {
                // FILTER ?x IN (a, b, c) / NOT IN (...): membership over a list.
                let left_val = self.evaluate_expression(left, binding)?;
                let candidates = self.collect_in_list(right, binding)?;
                let is_member = candidates
                    .iter()
                    .any(|candidate| self.terms_equal(&left_val, candidate));
                let result = matches!(op, crate::algebra::BinaryOperator::In) == is_member;
                Ok(crate::algebra::Term::Literal(crate::algebra::Literal {
                    value: result.to_string(),
                    language: None,
                    datatype: Some(oxirs_core::model::NamedNode::new_unchecked(
                        "http://www.w3.org/2001/XMLSchema#boolean",
                    )),
                }))
            }
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
                // String predicate functions (2 string arguments -> xsd:boolean).
                "contains" | "CONTAINS" => {
                    if args.len() == 2 {
                        let hay = self.evaluate_expression(&args[0], binding)?;
                        let needle = self.evaluate_expression(&args[1], binding)?;
                        self.string_predicate_function(&hay, &needle, "contains")
                    } else {
                        Err(anyhow::anyhow!("contains() requires exactly 2 arguments"))
                    }
                }
                "strstarts" | "STRSTARTS" => {
                    if args.len() == 2 {
                        let hay = self.evaluate_expression(&args[0], binding)?;
                        let needle = self.evaluate_expression(&args[1], binding)?;
                        self.string_predicate_function(&hay, &needle, "strstarts")
                    } else {
                        Err(anyhow::anyhow!("strstarts() requires exactly 2 arguments"))
                    }
                }
                "strends" | "STRENDS" => {
                    if args.len() == 2 {
                        let hay = self.evaluate_expression(&args[0], binding)?;
                        let needle = self.evaluate_expression(&args[1], binding)?;
                        self.string_predicate_function(&hay, &needle, "strends")
                    } else {
                        Err(anyhow::anyhow!("strends() requires exactly 2 arguments"))
                    }
                }
                // REGEX(text, pattern [, flags]) -> xsd:boolean.
                "regex" | "REGEX" => {
                    if (2..=3).contains(&args.len()) {
                        let text = self.evaluate_expression(&args[0], binding)?;
                        let pattern = self.evaluate_expression(&args[1], binding)?;
                        let flags = match args.get(2) {
                            Some(flag_expr) => Some(self.evaluate_expression(flag_expr, binding)?),
                            None => None,
                        };
                        self.regex_function(&text, &pattern, flags.as_ref())
                    } else {
                        Err(anyhow::anyhow!("regex() requires 2 or 3 arguments"))
                    }
                }
                // LANGMATCHES(language-tag, language-range) -> xsd:boolean.
                "langmatches" | "LANGMATCHES" => {
                    if args.len() == 2 {
                        let tag = self.evaluate_expression(&args[0], binding)?;
                        let range = self.evaluate_expression(&args[1], binding)?;
                        self.langmatches_function(&tag, &range)
                    } else {
                        Err(anyhow::anyhow!(
                            "langmatches() requires exactly 2 arguments"
                        ))
                    }
                }
                // COALESCE(expr, ...) returns the first argument that evaluates
                // without error. Arguments are evaluated lazily so an unbound
                // variable or type error in an earlier argument is skipped rather
                // than failing the whole call (SPARQL 1.1 §17.4.2.2).
                "coalesce" | "COALESCE" => {
                    for arg in args {
                        if let Ok(term) = self.evaluate_expression(arg, binding) {
                            return Ok(term);
                        }
                    }
                    Err(anyhow::anyhow!("COALESCE: no argument could be evaluated"))
                }
                _ => Err(anyhow::Error::new(super::types::UnknownFunctionError(
                    name.clone(),
                ))),
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
            // Substitute the current solution mapping INTO the inner pattern
            // (SPARQL 1.1 §18.2.1: EXISTS evaluates the pattern with the current
            // bindings substituted in). This yields the correct correlated
            // answer AND constrains the pattern — a substituted subject/object
            // turns a full store scan into a point lookup, so EXISTS does not
            // re-scan the whole store for every outer row. The previous
            // `Join(Values, pattern)` shape both scanned unconstrained and,
            // combined with the filter thread-local clear, fell into an
            // incorrect syntactic fallback whenever the inner pattern held a
            // FILTER.
            let substituted = substitute_algebra_binding(algebra, current_binding);
            match self.execute_serial(&substituted, dataset) {
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
    /// Collect the candidate list for an `IN` / `NOT IN` right operand.
    ///
    /// A list is encoded as a `list(...)`/`in(...)` function call; any other
    /// expression is treated as a single-element list.
    pub(super) fn collect_in_list(
        &self,
        expr: &crate::algebra::Expression,
        binding: &crate::algebra::Binding,
    ) -> Result<Vec<crate::algebra::Term>> {
        use crate::algebra::Expression;
        match expr {
            Expression::Function { name, args }
                if name.eq_ignore_ascii_case("list") || name.eq_ignore_ascii_case("in") =>
            {
                let mut out = Vec::with_capacity(args.len());
                for arg in args {
                    out.push(self.evaluate_expression(arg, binding)?);
                }
                Ok(out)
            }
            _ => Ok(vec![self.evaluate_expression(expr, binding)?]),
        }
    }

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
            BinaryOperator::In => {
                // Single-term membership: `left IN (right)`. Multi-element list
                // membership is handled in `evaluate_expression`, which has the
                // un-evaluated right operand available.
                let result = self.terms_equal(left, right);
                Ok(Term::Literal(crate::algebra::Literal {
                    value: result.to_string(),
                    language: None,
                    datatype: Some(oxirs_core::model::NamedNode::new_unchecked(
                        "http://www.w3.org/2001/XMLSchema#boolean",
                    )),
                }))
            }
            BinaryOperator::NotIn => {
                let result = !self.terms_equal(left, right);
                Ok(Term::Literal(crate::algebra::Literal {
                    value: result.to_string(),
                    language: None,
                    datatype: Some(oxirs_core::model::NamedNode::new_unchecked(
                        "http://www.w3.org/2001/XMLSchema#boolean",
                    )),
                }))
            }
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

    /// Extract the lexical string value of a term for string built-ins. A
    /// literal yields its lexical form; an IRI yields the IRI string. Any other
    /// term kind is a type error for these functions.
    pub(super) fn term_string_value(&self, term: &crate::algebra::Term) -> Result<String> {
        match term {
            crate::algebra::Term::Literal(lit) => Ok(lit.value.clone()),
            crate::algebra::Term::Iri(iri) => Ok(iri.as_str().to_string()),
            _ => Err(anyhow::anyhow!(
                "string function not applicable to this term type"
            )),
        }
    }

    /// Build an `xsd:boolean` term.
    pub(super) fn boolean_term(&self, value: bool) -> crate::algebra::Term {
        crate::algebra::Term::Literal(crate::algebra::Literal {
            value: value.to_string(),
            language: None,
            datatype: Some(oxirs_core::model::NamedNode::new_unchecked(
                "http://www.w3.org/2001/XMLSchema#boolean",
            )),
        })
    }

    /// CONTAINS / STRSTARTS / STRENDS: substring predicate over two string
    /// terms, returning an `xsd:boolean`. `kind` selects the predicate.
    pub(super) fn string_predicate_function(
        &self,
        haystack: &crate::algebra::Term,
        needle: &crate::algebra::Term,
        kind: &str,
    ) -> Result<crate::algebra::Term> {
        let hay = self.term_string_value(haystack)?;
        let needle = self.term_string_value(needle)?;
        let result = match kind {
            "contains" => hay.contains(&needle),
            "strstarts" => hay.starts_with(&needle),
            "strends" => hay.ends_with(&needle),
            other => return Err(anyhow::anyhow!("unknown string predicate: {other}")),
        };
        Ok(self.boolean_term(result))
    }

    /// Built-in `REGEX(text, pattern [, flags])` -> `xsd:boolean`. Supported
    /// flags: `i` (case-insensitive), `m` (multi-line), `s` (dot-all), `x`
    /// (extended/ignore-whitespace).
    pub(super) fn regex_function(
        &self,
        text: &crate::algebra::Term,
        pattern: &crate::algebra::Term,
        flags: Option<&crate::algebra::Term>,
    ) -> Result<crate::algebra::Term> {
        use regex::RegexBuilder;
        let text = self.term_string_value(text)?;
        let pattern = self.term_string_value(pattern)?;
        let mut builder = RegexBuilder::new(&pattern);
        if let Some(flags_term) = flags {
            let flags = self.term_string_value(flags_term)?;
            for flag in flags.chars() {
                match flag {
                    'i' => {
                        builder.case_insensitive(true);
                    }
                    'm' => {
                        builder.multi_line(true);
                    }
                    's' => {
                        builder.dot_matches_new_line(true);
                    }
                    'x' => {
                        builder.ignore_whitespace(true);
                    }
                    other => return Err(anyhow::anyhow!("unknown regex flag: {other}")),
                }
            }
        }
        let re = builder
            .build()
            .map_err(|e| anyhow::anyhow!("invalid regex pattern: {e}"))?;
        Ok(self.boolean_term(re.is_match(&text)))
    }

    /// Built-in `LANGMATCHES(language-tag, language-range)` -> `xsd:boolean`,
    /// implementing RFC 4647 basic-filtering matching: `*` matches any
    /// non-empty tag, otherwise the range matches the tag exactly or as a
    /// subtag-boundary prefix, case-insensitively.
    pub(super) fn langmatches_function(
        &self,
        tag: &crate::algebra::Term,
        range: &crate::algebra::Term,
    ) -> Result<crate::algebra::Term> {
        let tag = self.term_string_value(tag)?;
        let range = self.term_string_value(range)?;
        let matches = if range == "*" {
            !tag.is_empty()
        } else {
            let tag_l = tag.to_ascii_lowercase();
            let range_l = range.to_ascii_lowercase();
            tag_l == range_l
                || tag_l
                    .strip_prefix(&range_l)
                    .is_some_and(|rest| rest.starts_with('-'))
        };
        Ok(self.boolean_term(matches))
    }
}

/// Substitute a solution binding into an algebra sub-tree, replacing every
/// bound variable (in triple patterns, filter conditions, `BIND`/`GRAPH`, and
/// nested `EXISTS`) with its bound term.
///
/// This implements the SPARQL 1.1 §18.2.1 EXISTS/NOT EXISTS semantics: the
/// inner group graph pattern is evaluated with the current solution mapping
/// substituted in. Beyond correctness (correlated conditions such as
/// `FILTER(?e = ?outer)` now see the outer value), substitution constrains the
/// pattern so a bound subject/object becomes a point lookup instead of a full
/// store scan per outer row.
fn substitute_algebra_binding(
    algebra: &crate::algebra::Algebra,
    binding: &crate::algebra::Binding,
) -> crate::algebra::Algebra {
    use crate::algebra::Algebra as A;
    match algebra {
        A::Bgp(triples) => A::Bgp(
            triples
                .iter()
                .map(|t| substitute_triple_binding(t, binding))
                .collect(),
        ),
        A::Filter { pattern, condition } => A::Filter {
            pattern: Box::new(substitute_algebra_binding(pattern, binding)),
            condition: substitute_expression_binding(condition, binding),
        },
        A::Join { left, right } => A::Join {
            left: Box::new(substitute_algebra_binding(left, binding)),
            right: Box::new(substitute_algebra_binding(right, binding)),
        },
        A::LeftJoin {
            left,
            right,
            filter,
        } => A::LeftJoin {
            left: Box::new(substitute_algebra_binding(left, binding)),
            right: Box::new(substitute_algebra_binding(right, binding)),
            filter: filter
                .as_ref()
                .map(|f| substitute_expression_binding(f, binding)),
        },
        A::Union { left, right } => A::Union {
            left: Box::new(substitute_algebra_binding(left, binding)),
            right: Box::new(substitute_algebra_binding(right, binding)),
        },
        A::Minus { left, right } => A::Minus {
            left: Box::new(substitute_algebra_binding(left, binding)),
            right: Box::new(substitute_algebra_binding(right, binding)),
        },
        A::Graph { graph, pattern } => A::Graph {
            graph: substitute_term_binding(graph, binding),
            pattern: Box::new(substitute_algebra_binding(pattern, binding)),
        },
        A::Extend {
            pattern,
            variable,
            expr,
        } => A::Extend {
            pattern: Box::new(substitute_algebra_binding(pattern, binding)),
            variable: variable.clone(),
            expr: substitute_expression_binding(expr, binding),
        },
        // Other shapes (subquery projection/group/slice/values/…) are left
        // as-is: they are not produced inside an EXISTS group by this parser,
        // and their free variables still evaluate correctly (just unconstrained).
        other => other.clone(),
    }
}

fn substitute_triple_binding(
    triple: &crate::algebra::TriplePattern,
    binding: &crate::algebra::Binding,
) -> crate::algebra::TriplePattern {
    crate::algebra::TriplePattern::new(
        substitute_term_binding(&triple.subject, binding),
        substitute_term_binding(&triple.predicate, binding),
        substitute_term_binding(&triple.object, binding),
    )
}

fn substitute_term_binding(
    term: &crate::algebra::Term,
    binding: &crate::algebra::Binding,
) -> crate::algebra::Term {
    match term {
        crate::algebra::Term::Variable(var) => {
            binding.get(var).cloned().unwrap_or_else(|| term.clone())
        }
        _ => term.clone(),
    }
}

fn substitute_expression_binding(
    expr: &crate::algebra::Expression,
    binding: &crate::algebra::Binding,
) -> crate::algebra::Expression {
    use crate::algebra::Expression as E;
    match expr {
        E::Variable(var) => match binding.get(var) {
            Some(term) => term_to_expression(term).unwrap_or_else(|| expr.clone()),
            None => expr.clone(),
        },
        E::Binary { op, left, right } => E::Binary {
            op: op.clone(),
            left: Box::new(substitute_expression_binding(left, binding)),
            right: Box::new(substitute_expression_binding(right, binding)),
        },
        E::Unary { op, operand } => E::Unary {
            op: op.clone(),
            operand: Box::new(substitute_expression_binding(operand, binding)),
        },
        E::Function { name, args } => E::Function {
            name: name.clone(),
            args: args
                .iter()
                .map(|a| substitute_expression_binding(a, binding))
                .collect(),
        },
        E::Conditional {
            condition,
            then_expr,
            else_expr,
        } => E::Conditional {
            condition: Box::new(substitute_expression_binding(condition, binding)),
            then_expr: Box::new(substitute_expression_binding(then_expr, binding)),
            else_expr: Box::new(substitute_expression_binding(else_expr, binding)),
        },
        E::Exists(inner) => E::Exists(Box::new(substitute_algebra_binding(inner, binding))),
        E::NotExists(inner) => E::NotExists(Box::new(substitute_algebra_binding(inner, binding))),
        // BOUND(?v): if ?v is bound in the outer solution it is bound here too,
        // so fold it to a constant `true`; otherwise keep the check.
        E::Bound(var) => {
            if binding.contains_key(var) {
                E::Literal(crate::algebra::Literal {
                    value: "true".to_string(),
                    language: None,
                    datatype: Some(oxirs_core::model::NamedNode::new_unchecked(
                        "http://www.w3.org/2001/XMLSchema#boolean",
                    )),
                })
            } else {
                E::Bound(var.clone())
            }
        }
        E::Literal(_) | E::Iri(_) => expr.clone(),
    }
}

/// Lift a bound term into an expression operand. Blank nodes / property paths /
/// quoted triples have no expression form, so return `None` and leave the
/// original variable in place.
fn term_to_expression(term: &crate::algebra::Term) -> Option<crate::algebra::Expression> {
    match term {
        crate::algebra::Term::Iri(iri) => Some(crate::algebra::Expression::Iri(iri.clone())),
        crate::algebra::Term::Literal(lit) => {
            Some(crate::algebra::Expression::Literal(lit.clone()))
        }
        crate::algebra::Term::Variable(var) => {
            Some(crate::algebra::Expression::Variable(var.clone()))
        }
        _ => None,
    }
}
