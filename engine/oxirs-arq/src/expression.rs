//! Expression Evaluation System for SPARQL
//!
//! This module provides comprehensive expression evaluation capabilities
//! using the enhanced term system.

use crate::algebra::{
    Aggregate, BinaryOperator, Expression as AlgebraExpression, UnaryOperator, Variable,
};
use crate::extensions::{ExecutionContext as ExtContext, ExtensionRegistry};
use crate::term::{xsd, BindingContext, NumericValue, Term};
use anyhow::{anyhow, bail, Result};
use chrono::{Datelike, NaiveDate, NaiveDateTime};
use oxirs_core::model::NamedNode;
use std::collections::HashMap;
use std::sync::Arc;

/// Expression evaluator with full SPARQL 1.1 support
pub struct ExpressionEvaluator {
    /// Extension registry for custom functions
    extension_registry: Arc<ExtensionRegistry>,
    /// Current binding context
    binding_context: BindingContext,
}

impl ExpressionEvaluator {
    /// Create new expression evaluator
    pub fn new(extension_registry: Arc<ExtensionRegistry>) -> Self {
        Self {
            extension_registry,
            binding_context: BindingContext::new(),
        }
    }

    /// Create with existing binding context
    pub fn with_context(
        extension_registry: Arc<ExtensionRegistry>,
        context: BindingContext,
    ) -> Self {
        Self {
            extension_registry,
            binding_context: context,
        }
    }

    /// Evaluate expression to a term
    pub fn evaluate(&self, expr: &AlgebraExpression) -> Result<Term> {
        match expr {
            AlgebraExpression::Variable(var) => self
                .binding_context
                .get(var.as_str())
                .cloned()
                .ok_or_else(|| anyhow!("Unbound variable: ?{}", var)),

            AlgebraExpression::Literal(lit) => Ok(Term::from_algebra_term(
                &crate::algebra::Term::Literal(lit.clone()),
            )),

            AlgebraExpression::Iri(iri) => Ok(Term::iri(iri.as_str())),

            AlgebraExpression::Function { name, args } => self.evaluate_function(name, args),

            AlgebraExpression::Binary { op, left, right } => {
                let left_val = self.evaluate(left)?;
                let right_val = self.evaluate(right)?;
                self.evaluate_binary_op(op, &left_val, &right_val)
            }

            AlgebraExpression::Unary { op, expr } => {
                let val = self.evaluate(expr)?;
                self.evaluate_unary_op(op, &val)
            }

            AlgebraExpression::Conditional {
                condition,
                then_expr,
                else_expr,
            } => {
                let cond_val = self.evaluate(condition)?;
                if cond_val.effective_boolean_value()? {
                    self.evaluate(then_expr)
                } else {
                    self.evaluate(else_expr)
                }
            }

            AlgebraExpression::Bound(var) => Ok(Term::typed_literal(
                if self.binding_context.is_bound(var.as_str()) {
                    "true"
                } else {
                    "false"
                },
                xsd::BOOLEAN,
            )?),

            AlgebraExpression::Exists(_) | AlgebraExpression::NotExists(_) => {
                // These require subquery evaluation
                bail!("EXISTS/NOT EXISTS requires query executor context")
            }
        }
    }

    /// Evaluate function call
    fn evaluate_function(&self, name: &str, args: &[AlgebraExpression]) -> Result<Term> {
        // Evaluate arguments
        let arg_values: Vec<Term> = args
            .iter()
            .map(|arg| self.evaluate(arg))
            .collect::<Result<Vec<_>>>()?;

        // Check built-in functions first
        match name {
            // String functions
            "str" | "STR" => self.builtin_str(&arg_values),
            "strlen" | "STRLEN" => self.builtin_strlen(&arg_values),
            "substr" | "SUBSTR" => self.builtin_substr(&arg_values),
            "ucase" | "UCASE" => self.builtin_ucase(&arg_values),
            "lcase" | "LCASE" => self.builtin_lcase(&arg_values),
            "strstarts" | "STRSTARTS" => self.builtin_strstarts(&arg_values),
            "strends" | "STRENDS" => self.builtin_strends(&arg_values),
            "contains" | "CONTAINS" => self.builtin_contains(&arg_values),
            "concat" | "CONCAT" => self.builtin_concat(&arg_values),
            "replace" | "REPLACE" => self.builtin_replace(&arg_values),

            // Type checking functions
            "isIRI" | "isURI" => self.builtin_is_iri(&arg_values),
            "isBlank" | "isBLANK" => self.builtin_is_blank(&arg_values),
            "isLiteral" | "isLITERAL" => self.builtin_is_literal(&arg_values),
            "isNumeric" | "isNUMERIC" => self.builtin_is_numeric(&arg_values),

            // Numeric functions
            "abs" | "ABS" => self.builtin_abs(&arg_values),
            "round" | "ROUND" => self.builtin_round(&arg_values),
            "ceil" | "CEIL" => self.builtin_ceil(&arg_values),
            "floor" | "FLOOR" => self.builtin_floor(&arg_values),

            // Date/time functions
            "now" | "NOW" => self.builtin_now(&arg_values),
            "year" | "YEAR" => self.builtin_year(&arg_values),
            "month" | "MONTH" => self.builtin_month(&arg_values),
            "day" | "DAY" => self.builtin_day(&arg_values),

            // Logical functions
            "if" | "IF" => self.builtin_if(&arg_values),
            "coalesce" | "COALESCE" => self.builtin_coalesce(&arg_values),

            // Constructors
            "iri" | "IRI" | "uri" | "URI" => self.builtin_iri(&arg_values),
            "bnode" | "BNODE" => self.builtin_bnode(&arg_values),
            "strdt" | "STRDT" => self.builtin_strdt(&arg_values),
            "strlang" | "STRLANG" => self.builtin_strlang(&arg_values),

            // Check extension registry
            _ => bail!("Unknown function: {}", name),
        }
    }

    /// Evaluate binary operation
    fn evaluate_binary_op(&self, op: &BinaryOperator, left: &Term, right: &Term) -> Result<Term> {
        use BinaryOperator::*;

        match op {
            // Arithmetic operations
            Add | Subtract | Multiply | Divide => {
                let left_num = left.to_numeric()?;
                let right_num = right.to_numeric()?;
                let (left_prom, right_prom) = left_num.promote_with(&right_num);

                match (op, left_prom, right_prom) {
                    (Add, NumericValue::Integer(a), NumericValue::Integer(b)) => {
                        Ok(NumericValue::Integer(a + b).to_term())
                    }
                    (Add, NumericValue::Decimal(a), NumericValue::Decimal(b)) => {
                        Ok(NumericValue::Decimal(a + b).to_term())
                    }
                    (Add, NumericValue::Float(a), NumericValue::Float(b)) => {
                        Ok(NumericValue::Float(a + b).to_term())
                    }
                    (Add, NumericValue::Double(a), NumericValue::Double(b)) => {
                        Ok(NumericValue::Double(a + b).to_term())
                    }

                    (Subtract, NumericValue::Integer(a), NumericValue::Integer(b)) => {
                        Ok(NumericValue::Integer(a - b).to_term())
                    }
                    (Subtract, NumericValue::Decimal(a), NumericValue::Decimal(b)) => {
                        Ok(NumericValue::Decimal(a - b).to_term())
                    }
                    (Subtract, NumericValue::Float(a), NumericValue::Float(b)) => {
                        Ok(NumericValue::Float(a - b).to_term())
                    }
                    (Subtract, NumericValue::Double(a), NumericValue::Double(b)) => {
                        Ok(NumericValue::Double(a - b).to_term())
                    }

                    (Multiply, NumericValue::Integer(a), NumericValue::Integer(b)) => {
                        Ok(NumericValue::Integer(a * b).to_term())
                    }
                    (Multiply, NumericValue::Decimal(a), NumericValue::Decimal(b)) => {
                        Ok(NumericValue::Decimal(a * b).to_term())
                    }
                    (Multiply, NumericValue::Float(a), NumericValue::Float(b)) => {
                        Ok(NumericValue::Float(a * b).to_term())
                    }
                    (Multiply, NumericValue::Double(a), NumericValue::Double(b)) => {
                        Ok(NumericValue::Double(a * b).to_term())
                    }

                    (Divide, NumericValue::Integer(a), NumericValue::Integer(b)) => {
                        if b == 0 {
                            bail!("Division by zero")
                        }
                        Ok(NumericValue::Decimal(a as f64 / b as f64).to_term())
                    }
                    (Divide, NumericValue::Decimal(a), NumericValue::Decimal(b)) => {
                        if b == 0.0 {
                            bail!("Division by zero")
                        }
                        Ok(NumericValue::Decimal(a / b).to_term())
                    }
                    (Divide, NumericValue::Float(a), NumericValue::Float(b)) => {
                        if b == 0.0 {
                            bail!("Division by zero")
                        }
                        Ok(NumericValue::Float(a / b).to_term())
                    }
                    (Divide, NumericValue::Double(a), NumericValue::Double(b)) => {
                        if b == 0.0 {
                            bail!("Division by zero")
                        }
                        Ok(NumericValue::Double(a / b).to_term())
                    }

                    _ => unreachable!(),
                }
            }

            // Comparison operations
            Equal => Ok(self.bool_term(left == right)),
            NotEqual => Ok(self.bool_term(left != right)),
            Less => Ok(self.bool_term(left < right)),
            LessEqual => Ok(self.bool_term(left <= right)),
            Greater => Ok(self.bool_term(left > right)),
            GreaterEqual => Ok(self.bool_term(left >= right)),

            // Logical operations
            And => {
                let left_bool = left.effective_boolean_value()?;
                let right_bool = right.effective_boolean_value()?;
                Ok(self.bool_term(left_bool && right_bool))
            }
            Or => {
                let left_bool = left.effective_boolean_value()?;
                let right_bool = right.effective_boolean_value()?;
                Ok(self.bool_term(left_bool || right_bool))
            }

            // RDF term equality
            SameTerm => Ok(self.bool_term(left == right)),

            // Set membership
            In | NotIn => {
                bail!("IN/NOT IN requires list context")
            }
        }
    }

    /// Evaluate unary operation
    fn evaluate_unary_op(&self, op: &UnaryOperator, term: &Term) -> Result<Term> {
        use UnaryOperator::*;

        match op {
            Not => {
                let bool_val = term.effective_boolean_value()?;
                Ok(self.bool_term(!bool_val))
            }
            Plus => {
                let num = term.to_numeric()?;
                Ok(num.to_term())
            }
            Minus => {
                let num = term.to_numeric()?;
                match num {
                    NumericValue::Integer(i) => Ok(NumericValue::Integer(-i).to_term()),
                    NumericValue::Decimal(d) => Ok(NumericValue::Decimal(-d).to_term()),
                    NumericValue::Float(f) => Ok(NumericValue::Float(-f).to_term()),
                    NumericValue::Double(d) => Ok(NumericValue::Double(-d).to_term()),
                }
            }
            IsIri => Ok(self.bool_term(term.is_iri())),
            IsBlank => Ok(self.bool_term(term.is_blank_node())),
            IsLiteral => Ok(self.bool_term(term.is_literal())),
            IsNumeric => match term {
                Term::Literal(lit) => Ok(self.bool_term(lit.is_numeric())),
                _ => Ok(self.bool_term(false)),
            },
        }
    }

    // Built-in function implementations

    fn builtin_str(&self, args: &[Term]) -> Result<Term> {
        if args.len() != 1 {
            bail!("STR expects 1 argument");
        }

        let str_val = match &args[0] {
            Term::Iri(iri) => iri.clone(),
            Term::Literal(lit) => lit.lexical_form.clone(),
            Term::BlankNode(id) => format!("_:{}", id),
            Term::Variable(var) => format!("?{}", var),
            Term::QuotedTriple(triple) => {
                format!("<<{} {} {}>>", triple.subject, triple.predicate, triple.object)
            }
            Term::PropertyPath(path) => format!("{}", path),
        };

        Ok(Term::literal(&str_val))
    }

    fn builtin_strlen(&self, args: &[Term]) -> Result<Term> {
        if args.len() != 1 {
            bail!("STRLEN expects 1 argument");
        }

        let str_term = self.builtin_str(args)?;
        if let Term::Literal(lit) = str_term {
            let len = lit.lexical_form.chars().count() as i64;
            Ok(Term::typed_literal(&len.to_string(), xsd::INTEGER)?)
        } else {
            unreachable!()
        }
    }

    fn builtin_substr(&self, args: &[Term]) -> Result<Term> {
        if args.len() < 2 || args.len() > 3 {
            bail!("SUBSTR expects 2 or 3 arguments");
        }

        let str_term = self.builtin_str(&[args[0].clone()])?;
        let start = args[1].to_numeric()?.to_term();

        if let (Term::Literal(str_lit), Term::Literal(start_lit)) = (str_term, start) {
            let start_pos = start_lit
                .lexical_form
                .parse::<usize>()
                .map_err(|_| anyhow!("Invalid start position"))?;

            let chars: Vec<char> = str_lit.lexical_form.chars().collect();
            let result = if args.len() == 3 {
                let len = args[2].to_numeric()?.to_term();
                if let Term::Literal(len_lit) = len {
                    let length = len_lit
                        .lexical_form
                        .parse::<usize>()
                        .map_err(|_| anyhow!("Invalid length"))?;
                    chars
                        .iter()
                        .skip(start_pos.saturating_sub(1))
                        .take(length)
                        .collect::<String>()
                } else {
                    unreachable!()
                }
            } else {
                chars
                    .iter()
                    .skip(start_pos.saturating_sub(1))
                    .collect::<String>()
            };

            Ok(Term::literal(&result))
        } else {
            unreachable!()
        }
    }

    fn builtin_ucase(&self, args: &[Term]) -> Result<Term> {
        if args.len() != 1 {
            bail!("UCASE expects 1 argument");
        }

        let str_term = self.builtin_str(args)?;
        if let Term::Literal(lit) = str_term {
            Ok(Term::literal(&lit.lexical_form.to_uppercase()))
        } else {
            unreachable!()
        }
    }

    fn builtin_lcase(&self, args: &[Term]) -> Result<Term> {
        if args.len() != 1 {
            bail!("LCASE expects 1 argument");
        }

        let str_term = self.builtin_str(args)?;
        if let Term::Literal(lit) = str_term {
            Ok(Term::literal(&lit.lexical_form.to_lowercase()))
        } else {
            unreachable!()
        }
    }

    fn builtin_strstarts(&self, args: &[Term]) -> Result<Term> {
        if args.len() != 2 {
            bail!("STRSTARTS expects 2 arguments");
        }

        let str1 = self.builtin_str(&[args[0].clone()])?;
        let str2 = self.builtin_str(&[args[1].clone()])?;

        if let (Term::Literal(lit1), Term::Literal(lit2)) = (str1, str2) {
            Ok(self.bool_term(lit1.lexical_form.starts_with(&lit2.lexical_form)))
        } else {
            unreachable!()
        }
    }

    fn builtin_strends(&self, args: &[Term]) -> Result<Term> {
        if args.len() != 2 {
            bail!("STRENDS expects 2 arguments");
        }

        let str1 = self.builtin_str(&[args[0].clone()])?;
        let str2 = self.builtin_str(&[args[1].clone()])?;

        if let (Term::Literal(lit1), Term::Literal(lit2)) = (str1, str2) {
            Ok(self.bool_term(lit1.lexical_form.ends_with(&lit2.lexical_form)))
        } else {
            unreachable!()
        }
    }

    fn builtin_contains(&self, args: &[Term]) -> Result<Term> {
        if args.len() != 2 {
            bail!("CONTAINS expects 2 arguments");
        }

        let str1 = self.builtin_str(&[args[0].clone()])?;
        let str2 = self.builtin_str(&[args[1].clone()])?;

        if let (Term::Literal(lit1), Term::Literal(lit2)) = (str1, str2) {
            Ok(self.bool_term(lit1.lexical_form.contains(&lit2.lexical_form)))
        } else {
            unreachable!()
        }
    }

    fn builtin_concat(&self, args: &[Term]) -> Result<Term> {
        let mut result = String::new();

        for arg in args {
            let str_term = self.builtin_str(&[arg.clone()])?;
            if let Term::Literal(lit) = str_term {
                result.push_str(&lit.lexical_form);
            }
        }

        Ok(Term::literal(&result))
    }

    fn builtin_replace(&self, args: &[Term]) -> Result<Term> {
        if args.len() < 3 || args.len() > 4 {
            bail!("REPLACE expects 3 or 4 arguments");
        }

        let input = self.builtin_str(&[args[0].clone()])?;
        let pattern = self.builtin_str(&[args[1].clone()])?;
        let replacement = self.builtin_str(&[args[2].clone()])?;

        if let (Term::Literal(input_lit), Term::Literal(pattern_lit), Term::Literal(repl_lit)) =
            (input, pattern, replacement)
        {
            let result = if args.len() == 4 {
                // Handle regex flags in 4th argument
                let flags = self.builtin_str(&[args[3].clone()])?;
                if let Term::Literal(flags_lit) = flags {
                    use regex::RegexBuilder;

                    let mut builder = RegexBuilder::new(&pattern_lit.lexical_form);

                    // Parse flags
                    for flag in flags_lit.lexical_form.chars() {
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
                            _ => bail!("Unknown regex flag: {}", flag),
                        }
                    }

                    match builder.build() {
                        Ok(regex) => regex
                            .replace_all(&input_lit.lexical_form, repl_lit.lexical_form.as_str())
                            .to_string(),
                        Err(e) => bail!("Invalid regex pattern: {}", e),
                    }
                } else {
                    unreachable!()
                }
            } else {
                // Simple string replacement without regex
                input_lit
                    .lexical_form
                    .replace(&pattern_lit.lexical_form, &repl_lit.lexical_form)
            };

            Ok(Term::literal(&result))
        } else {
            unreachable!()
        }
    }

    fn builtin_is_iri(&self, args: &[Term]) -> Result<Term> {
        if args.len() != 1 {
            bail!("isIRI expects 1 argument");
        }
        Ok(self.bool_term(args[0].is_iri()))
    }

    fn builtin_is_blank(&self, args: &[Term]) -> Result<Term> {
        if args.len() != 1 {
            bail!("isBlank expects 1 argument");
        }
        Ok(self.bool_term(args[0].is_blank_node()))
    }

    fn builtin_is_literal(&self, args: &[Term]) -> Result<Term> {
        if args.len() != 1 {
            bail!("isLiteral expects 1 argument");
        }
        Ok(self.bool_term(args[0].is_literal()))
    }

    fn builtin_is_numeric(&self, args: &[Term]) -> Result<Term> {
        if args.len() != 1 {
            bail!("isNumeric expects 1 argument");
        }

        let is_numeric = match &args[0] {
            Term::Literal(lit) => lit.is_numeric(),
            _ => false,
        };

        Ok(self.bool_term(is_numeric))
    }

    fn builtin_abs(&self, args: &[Term]) -> Result<Term> {
        if args.len() != 1 {
            bail!("ABS expects 1 argument");
        }

        let num = args[0].to_numeric()?;
        match num {
            NumericValue::Integer(i) => Ok(NumericValue::Integer(i.abs()).to_term()),
            NumericValue::Decimal(d) => Ok(NumericValue::Decimal(d.abs()).to_term()),
            NumericValue::Float(f) => Ok(NumericValue::Float(f.abs()).to_term()),
            NumericValue::Double(d) => Ok(NumericValue::Double(d.abs()).to_term()),
        }
    }

    fn builtin_round(&self, args: &[Term]) -> Result<Term> {
        if args.len() != 1 {
            bail!("ROUND expects 1 argument");
        }

        let num = args[0].to_numeric()?;
        match num {
            NumericValue::Integer(i) => Ok(NumericValue::Integer(i).to_term()),
            NumericValue::Decimal(d) => Ok(NumericValue::Decimal(d.round()).to_term()),
            NumericValue::Float(f) => Ok(NumericValue::Float(f.round()).to_term()),
            NumericValue::Double(d) => Ok(NumericValue::Double(d.round()).to_term()),
        }
    }

    fn builtin_ceil(&self, args: &[Term]) -> Result<Term> {
        if args.len() != 1 {
            bail!("CEIL expects 1 argument");
        }

        let num = args[0].to_numeric()?;
        match num {
            NumericValue::Integer(i) => Ok(NumericValue::Integer(i).to_term()),
            NumericValue::Decimal(d) => Ok(NumericValue::Decimal(d.ceil()).to_term()),
            NumericValue::Float(f) => Ok(NumericValue::Float(f.ceil()).to_term()),
            NumericValue::Double(d) => Ok(NumericValue::Double(d.ceil()).to_term()),
        }
    }

    fn builtin_floor(&self, args: &[Term]) -> Result<Term> {
        if args.len() != 1 {
            bail!("FLOOR expects 1 argument");
        }

        let num = args[0].to_numeric()?;
        match num {
            NumericValue::Integer(i) => Ok(NumericValue::Integer(i).to_term()),
            NumericValue::Decimal(d) => Ok(NumericValue::Decimal(d.floor()).to_term()),
            NumericValue::Float(f) => Ok(NumericValue::Float(f.floor()).to_term()),
            NumericValue::Double(d) => Ok(NumericValue::Double(d.floor()).to_term()),
        }
    }

    fn builtin_now(&self, _args: &[Term]) -> Result<Term> {
        let now = chrono::Utc::now();
        Term::typed_literal(&now.to_rfc3339(), xsd::DATE_TIME)
    }

    fn builtin_year(&self, args: &[Term]) -> Result<Term> {
        if args.len() != 1 {
            bail!("YEAR expects 1 argument");
        }

        match &args[0] {
            Term::Literal(lit) => {
                match lit.datatype.as_str() {
                    xsd::DATE | xsd::DATE_TIME | xsd::DATE_TIME_STAMP => {
                        // Parse ISO 8601 date/datetime
                        if let Ok(datetime) =
                            chrono::DateTime::parse_from_rfc3339(&lit.lexical_form)
                        {
                            Term::typed_literal(
                                &datetime.year().to_string(),
                                "http://www.w3.org/2001/XMLSchema#integer",
                            )
                        } else if let Ok(date) =
                            chrono::NaiveDate::parse_from_str(&lit.lexical_form, "%Y-%m-%d")
                        {
                            Term::typed_literal(
                                &date.year().to_string(),
                                "http://www.w3.org/2001/XMLSchema#integer",
                            )
                        } else if let Ok(datetime) = chrono::NaiveDateTime::parse_from_str(
                            &lit.lexical_form,
                            "%Y-%m-%dT%H:%M:%S",
                        ) {
                            Term::typed_literal(
                                &datetime.year().to_string(),
                                "http://www.w3.org/2001/XMLSchema#integer",
                            )
                        } else {
                            bail!("Invalid date/dateTime format: {}", lit.lexical_form)
                        }
                    }
                    _ => bail!("YEAR function can only be applied to date/dateTime literals"),
                }
            }
            _ => bail!("YEAR function requires a literal argument"),
        }
    }

    fn builtin_month(&self, args: &[Term]) -> Result<Term> {
        if args.len() != 1 {
            bail!("MONTH expects 1 argument");
        }

        match &args[0] {
            Term::Literal(lit) => {
                match lit.datatype.as_str() {
                    xsd::DATE | xsd::DATE_TIME | xsd::DATE_TIME_STAMP => {
                        // Parse ISO 8601 date/datetime
                        if let Ok(datetime) =
                            chrono::DateTime::parse_from_rfc3339(&lit.lexical_form)
                        {
                            Term::typed_literal(
                                &datetime.month().to_string(),
                                "http://www.w3.org/2001/XMLSchema#integer",
                            )
                        } else if let Ok(date) =
                            chrono::NaiveDate::parse_from_str(&lit.lexical_form, "%Y-%m-%d")
                        {
                            Term::typed_literal(
                                &date.month().to_string(),
                                "http://www.w3.org/2001/XMLSchema#integer",
                            )
                        } else if let Ok(datetime) = chrono::NaiveDateTime::parse_from_str(
                            &lit.lexical_form,
                            "%Y-%m-%dT%H:%M:%S",
                        ) {
                            Term::typed_literal(
                                &datetime.month().to_string(),
                                "http://www.w3.org/2001/XMLSchema#integer",
                            )
                        } else {
                            bail!("Invalid date/dateTime format: {}", lit.lexical_form)
                        }
                    }
                    _ => bail!("MONTH function can only be applied to date/dateTime literals"),
                }
            }
            _ => bail!("MONTH function requires a literal argument"),
        }
    }

    fn builtin_day(&self, args: &[Term]) -> Result<Term> {
        if args.len() != 1 {
            bail!("DAY expects 1 argument");
        }

        match &args[0] {
            Term::Literal(lit) => {
                match lit.datatype.as_str() {
                    xsd::DATE | xsd::DATE_TIME | xsd::DATE_TIME_STAMP => {
                        // Parse ISO 8601 date/datetime
                        if let Ok(datetime) =
                            chrono::DateTime::parse_from_rfc3339(&lit.lexical_form)
                        {
                            Term::typed_literal(
                                &datetime.day().to_string(),
                                "http://www.w3.org/2001/XMLSchema#integer",
                            )
                        } else if let Ok(date) =
                            chrono::NaiveDate::parse_from_str(&lit.lexical_form, "%Y-%m-%d")
                        {
                            Term::typed_literal(
                                &date.day().to_string(),
                                "http://www.w3.org/2001/XMLSchema#integer",
                            )
                        } else if let Ok(datetime) = chrono::NaiveDateTime::parse_from_str(
                            &lit.lexical_form,
                            "%Y-%m-%dT%H:%M:%S",
                        ) {
                            Term::typed_literal(
                                &datetime.day().to_string(),
                                "http://www.w3.org/2001/XMLSchema#integer",
                            )
                        } else {
                            bail!("Invalid date/dateTime format: {}", lit.lexical_form)
                        }
                    }
                    _ => bail!("DAY function can only be applied to date/dateTime literals"),
                }
            }
            _ => bail!("DAY function requires a literal argument"),
        }
    }

    fn builtin_if(&self, args: &[Term]) -> Result<Term> {
        if args.len() != 3 {
            bail!("IF expects 3 arguments");
        }

        if args[0].effective_boolean_value()? {
            Ok(args[1].clone())
        } else {
            Ok(args[2].clone())
        }
    }

    fn builtin_coalesce(&self, args: &[Term]) -> Result<Term> {
        for arg in args {
            if !arg.is_variable() {
                return Ok(arg.clone());
            }
        }
        bail!("COALESCE: all arguments are unbound")
    }

    fn builtin_iri(&self, args: &[Term]) -> Result<Term> {
        if args.len() != 1 {
            bail!("IRI expects 1 argument");
        }

        let str_val = self.builtin_str(&[args[0].clone()])?;
        if let Term::Literal(lit) = str_val {
            Ok(Term::iri(&lit.lexical_form))
        } else {
            unreachable!()
        }
    }

    fn builtin_bnode(&self, args: &[Term]) -> Result<Term> {
        if args.is_empty() {
            // Generate new blank node
            Ok(Term::blank_node(&format!("b{}", uuid::Uuid::new_v4())))
        } else if args.len() == 1 {
            let str_val = self.builtin_str(&[args[0].clone()])?;
            if let Term::Literal(lit) = str_val {
                Ok(Term::blank_node(&lit.lexical_form))
            } else {
                unreachable!()
            }
        } else {
            bail!("BNODE expects 0 or 1 argument");
        }
    }

    fn builtin_strdt(&self, args: &[Term]) -> Result<Term> {
        if args.len() != 2 {
            bail!("STRDT expects 2 arguments");
        }

        let str_val = self.builtin_str(&[args[0].clone()])?;
        if let (Term::Literal(lit), Term::Iri(dt)) = (str_val, &args[1]) {
            Term::typed_literal(&lit.lexical_form, dt)
        } else {
            bail!("STRDT expects string and IRI arguments")
        }
    }

    fn builtin_strlang(&self, args: &[Term]) -> Result<Term> {
        if args.len() != 2 {
            bail!("STRLANG expects 2 arguments");
        }

        let str_val = self.builtin_str(&[args[0].clone()])?;
        let lang_val = self.builtin_str(&[args[1].clone()])?;

        if let (Term::Literal(str_lit), Term::Literal(lang_lit)) = (str_val, lang_val) {
            Ok(Term::lang_literal(
                &str_lit.lexical_form,
                &lang_lit.lexical_form,
            ))
        } else {
            unreachable!()
        }
    }

    /// Helper to create boolean term
    fn bool_term(&self, value: bool) -> Term {
        Term::typed_literal(if value { "true" } else { "false" }, xsd::BOOLEAN).unwrap()
    }

    /// Get mutable binding context
    pub fn binding_context_mut(&mut self) -> &mut BindingContext {
        &mut self.binding_context
    }

    /// Get binding context
    pub fn binding_context(&self) -> &BindingContext {
        &self.binding_context
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expression_evaluation() {
        let registry = ExtensionRegistry::new();
        let mut evaluator = ExpressionEvaluator::new(Arc::new(registry));

        // Bind some variables
        evaluator
            .binding_context_mut()
            .bind("x", Term::typed_literal("42", xsd::INTEGER).unwrap());
        evaluator
            .binding_context_mut()
            .bind("y", Term::typed_literal("3.14", xsd::DOUBLE).unwrap());

        // Test variable evaluation
        let expr = AlgebraExpression::Variable(Variable::new("x").unwrap());
        let result = evaluator.evaluate(&expr).unwrap();
        assert_eq!(result, Term::typed_literal("42", xsd::INTEGER).unwrap());

        // Test arithmetic
        let add_expr = AlgebraExpression::Binary {
            op: BinaryOperator::Add,
            left: Box::new(AlgebraExpression::Variable(Variable::new("x").unwrap())),
            right: Box::new(AlgebraExpression::Literal(crate::algebra::Literal {
                value: "8".to_string(),
                language: None,
                datatype: Some(NamedNode::new(xsd::INTEGER).unwrap()),
            })),
        };

        let result = evaluator.evaluate(&add_expr).unwrap();
        assert_eq!(result, Term::typed_literal("50", xsd::INTEGER).unwrap());
    }

    #[test]
    fn test_builtin_functions() {
        let registry = ExtensionRegistry::new();
        let evaluator = ExpressionEvaluator::new(Arc::new(registry));

        // Test STR function
        let iri_term = Term::iri("http://example.org/foo");
        let result = evaluator.builtin_str(&[iri_term]).unwrap();
        assert_eq!(result, Term::literal("http://example.org/foo"));

        // Test STRLEN
        let str_term = Term::literal("hello");
        let result = evaluator.builtin_strlen(&[str_term]).unwrap();
        assert_eq!(result, Term::typed_literal("5", xsd::INTEGER).unwrap());

        // Test UCASE
        let str_term = Term::literal("hello");
        let result = evaluator.builtin_ucase(&[str_term]).unwrap();
        assert_eq!(result, Term::literal("HELLO"));
    }
}
