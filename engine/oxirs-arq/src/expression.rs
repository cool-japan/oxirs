//! Expression Evaluation System for SPARQL
//!
//! This module provides comprehensive expression evaluation capabilities
//! using the enhanced term system.

use crate::algebra::{BinaryOperator, Expression as AlgebraExpression, UnaryOperator};
use crate::extensions::ExtensionRegistry;
use crate::term::{xsd, BindingContext, NumericValue, Term};
use anyhow::{anyhow, bail, Result};
use chrono::{Datelike, Timelike};
use std::sync::Arc;

/// Expression evaluator with full SPARQL 1.1 support
pub struct ExpressionEvaluator {
    /// Extension registry for custom functions
    #[allow(dead_code)]
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
                .ok_or_else(|| anyhow!("Unbound variable: ?{var}")),

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

            AlgebraExpression::Unary { op, operand } => {
                let val = self.evaluate(operand)?;
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
            "strbefore" | "STRBEFORE" => self.builtin_strbefore(&arg_values),
            "strafter" | "STRAFTER" => self.builtin_strafter(&arg_values),
            "encode_for_uri" | "ENCODE_FOR_URI" => self.builtin_encode_for_uri(&arg_values),
            "regex" | "REGEX" => self.builtin_regex(&arg_values),

            // RDF term / literal accessors
            "lang" | "LANG" => self.builtin_lang(&arg_values),
            "datatype" | "DATATYPE" => self.builtin_datatype(&arg_values),
            "langmatches" | "LANGMATCHES" => self.builtin_lang_matches(&arg_values),
            "sameterm" | "sameTerm" | "SAMETERM" => self.builtin_same_term(&arg_values),

            // Hash functions
            "md5" | "MD5" => self.builtin_md5(&arg_values),
            "sha1" | "SHA1" => self.builtin_sha1(&arg_values),
            "sha256" | "SHA256" => self.builtin_sha256(&arg_values),
            "sha384" | "SHA384" => self.builtin_sha384(&arg_values),
            "sha512" | "SHA512" => self.builtin_sha512(&arg_values),

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
            "hours" | "HOURS" => self.builtin_hours(&arg_values),
            "minutes" | "MINUTES" => self.builtin_minutes(&arg_values),
            "seconds" | "SECONDS" => self.builtin_seconds(&arg_values),
            "timezone" | "TIMEZONE" => self.builtin_timezone(&arg_values),
            "tz" | "TZ" => self.builtin_tz(&arg_values),

            // Identifier generators
            "uuid" | "UUID" => self.builtin_uuid(&arg_values),
            "struuid" | "STRUUID" => self.builtin_struuid(&arg_values),

            // Logical functions
            "if" | "IF" => self.builtin_if(&arg_values),
            "coalesce" | "COALESCE" => self.builtin_coalesce(&arg_values),

            // Constructors
            "iri" | "IRI" | "uri" | "URI" => self.builtin_iri(&arg_values),
            "bnode" | "BNODE" => self.builtin_bnode(&arg_values),
            "strdt" | "STRDT" => self.builtin_strdt(&arg_values),
            "strlang" | "STRLANG" => self.builtin_strlang(&arg_values),

            // Fall back to the extension registry for any name that is not a
            // built-in — user-registered custom functions are keyed by their
            // full IRI. Only after that lookup fails do we report the function
            // as unknown (fail-loud: never silently return an empty/wrong value).
            _ => self.evaluate_extension_function(name, &arg_values),
        }
    }

    /// Look a non-built-in function name up in the extension registry (keyed by
    /// IRI) and evaluate it. Returns an explicit error when the name is not a
    /// registered extension, so an unknown function never silently succeeds.
    fn evaluate_extension_function(&self, name: &str, arg_values: &[Term]) -> Result<Term> {
        if !self.extension_registry.has_function(name)? {
            bail!("Unknown function: {name}");
        }
        let args: Vec<crate::extensions::Value> = arg_values
            .iter()
            .map(term_to_extension_value)
            .collect::<Result<Vec<_>>>()?;
        let context = crate::extensions::ExecutionContext::default();
        let value = self
            .extension_registry
            .execute_function(name, &args, &context)?;
        extension_value_to_term(&value)
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
            Term::BlankNode(id) => format!("_:{id}"),
            Term::Variable(var) => format!("?{var}"),
            Term::QuotedTriple(triple) => {
                format!(
                    "<<{} {} {}>>",
                    triple.subject, triple.predicate, triple.object
                )
            }
            Term::PropertyPath(path) => format!("{path}"),
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
            let str_term = self.builtin_str(std::slice::from_ref(arg))?;
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
            // SPARQL REPLACE is fn:replace: the pattern is ALWAYS a regular
            // expression (never a literal string) and `$N` in the replacement
            // refers to capture group N. Compile the pattern honoring any flags.
            let mut builder = regex::RegexBuilder::new(&pattern_lit.lexical_form);
            if args.len() == 4 {
                let flags = self.builtin_str(&[args[3].clone()])?;
                if let Term::Literal(flags_lit) = flags {
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
                            _ => bail!("Unknown regex flag: {flag}"),
                        }
                    }
                }
            }
            let regex = builder
                .build()
                .map_err(|e| anyhow!("Invalid REPLACE pattern: {e}"))?;
            let result = regex
                .replace_all(&input_lit.lexical_form, repl_lit.lexical_form.as_str())
                .to_string();

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
                            bail!(
                                "Invalid date/dateTime format: {lex}",
                                lex = lit.lexical_form
                            )
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
                            bail!(
                                "Invalid date/dateTime format: {lex}",
                                lex = lit.lexical_form
                            )
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
                            bail!(
                                "Invalid date/dateTime format: {lex}",
                                lex = lit.lexical_form
                            )
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
            Ok(Term::blank_node(&format!(
                "b{uuid}",
                uuid = uuid::Uuid::new_v4()
            )))
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

    /// `LANG(lit)` — the language tag of a literal (lower-cased is not required
    /// by SPARQL; the tag is returned verbatim), or the empty string for a
    /// non-language-tagged literal.
    fn builtin_lang(&self, args: &[Term]) -> Result<Term> {
        if args.len() != 1 {
            bail!("LANG expects 1 argument");
        }
        match &args[0] {
            Term::Literal(lit) => Ok(Term::literal(lit.language_tag.as_deref().unwrap_or(""))),
            _ => bail!("LANG requires a literal argument"),
        }
    }

    /// `DATATYPE(lit)` — the datatype IRI of a literal. A language-tagged literal
    /// has datatype `rdf:langString`; a plain literal has `xsd:string`.
    fn builtin_datatype(&self, args: &[Term]) -> Result<Term> {
        if args.len() != 1 {
            bail!("DATATYPE expects 1 argument");
        }
        match &args[0] {
            Term::Literal(lit) => Ok(Term::iri(&lit.datatype)),
            _ => bail!("DATATYPE requires a literal argument"),
        }
    }

    /// `langMatches(tag, range)` — RFC 4647 basic language-range matching. The
    /// range `"*"` matches any non-empty tag; otherwise the tag must equal the
    /// range or extend it with a `-`-separated subtag, compared case-insensitively.
    fn builtin_lang_matches(&self, args: &[Term]) -> Result<Term> {
        if args.len() != 2 {
            bail!("langMatches expects 2 arguments");
        }
        let tag = self.string_value(&args[0])?.to_lowercase();
        let range = self.string_value(&args[1])?.to_lowercase();
        let matches = if range == "*" {
            !tag.is_empty()
        } else {
            tag == range || tag.starts_with(&format!("{range}-"))
        };
        Ok(self.bool_term(matches))
    }

    /// `sameTerm(a, b)` — RDF term identity (not value equality): two terms are
    /// the same iff they are syntactically identical.
    fn builtin_same_term(&self, args: &[Term]) -> Result<Term> {
        if args.len() != 2 {
            bail!("sameTerm expects 2 arguments");
        }
        Ok(self.bool_term(args[0] == args[1]))
    }

    /// `REGEX(text, pattern, flags?)` — true iff `text` matches the regular
    /// expression `pattern`. Supported flags: `i` (case-insensitive), `m`
    /// (multi-line), `s` (dot matches newline), `x` (ignore whitespace). An
    /// invalid pattern is a runtime error, never a silent `false`.
    fn builtin_regex(&self, args: &[Term]) -> Result<Term> {
        if args.len() < 2 || args.len() > 3 {
            bail!("REGEX expects 2 or 3 arguments");
        }
        let text = self.string_value(&args[0])?;
        let pattern = self.string_value(&args[1])?;
        let mut builder = regex::RegexBuilder::new(&pattern);
        if args.len() == 3 {
            let flags = self.string_value(&args[2])?;
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
                    other => bail!("Unknown REGEX flag: {other}"),
                }
            }
        }
        let regex = builder
            .build()
            .map_err(|e| anyhow!("Invalid REGEX pattern: {e}"))?;
        Ok(self.bool_term(regex.is_match(&text)))
    }

    /// `STRBEFORE(arg1, arg2)` — the substring of `arg1` that precedes the first
    /// occurrence of `arg2`. Returns the empty string when `arg2` is not found;
    /// when `arg2` is the empty string the result is the empty string. The
    /// language tag of `arg1` is preserved on a non-empty result.
    fn builtin_strbefore(&self, args: &[Term]) -> Result<Term> {
        if args.len() != 2 {
            bail!("STRBEFORE expects 2 arguments");
        }
        let (haystack, lang) = self.literal_parts(&args[0])?;
        let needle = self.string_value(&args[1])?;
        if needle.is_empty() {
            return Ok(Term::literal(""));
        }
        match haystack.find(&needle) {
            Some(idx) => Ok(self.make_str_literal(&haystack[..idx], &lang)),
            None => Ok(Term::literal("")),
        }
    }

    /// `STRAFTER(arg1, arg2)` — the substring of `arg1` following the first
    /// occurrence of `arg2`. Returns the empty string when `arg2` is not found;
    /// when `arg2` is empty the result is `arg1` unchanged.
    fn builtin_strafter(&self, args: &[Term]) -> Result<Term> {
        if args.len() != 2 {
            bail!("STRAFTER expects 2 arguments");
        }
        let (haystack, lang) = self.literal_parts(&args[0])?;
        let needle = self.string_value(&args[1])?;
        if needle.is_empty() {
            return Ok(self.make_str_literal(&haystack, &lang));
        }
        match haystack.find(&needle) {
            Some(idx) => {
                let start = idx + needle.len();
                Ok(self.make_str_literal(&haystack[start..], &lang))
            }
            None => Ok(Term::literal("")),
        }
    }

    /// `ENCODE_FOR_URI(str)` — percent-encode every character that is not an
    /// unreserved URI character (`ALPHA / DIGIT / '-' / '.' / '_' / '~'`).
    fn builtin_encode_for_uri(&self, args: &[Term]) -> Result<Term> {
        if args.len() != 1 {
            bail!("ENCODE_FOR_URI expects 1 argument");
        }
        let input = self.string_value(&args[0])?;
        let mut out = String::with_capacity(input.len());
        for byte in input.bytes() {
            let c = byte as char;
            if c.is_ascii_alphanumeric() || matches!(c, '-' | '.' | '_' | '~') {
                out.push(c);
            } else {
                out.push_str(&format!("%{byte:02X}"));
            }
        }
        Ok(Term::literal(&out))
    }

    /// `HOURS(dt)` — the hour component (0-23) of a dateTime.
    fn builtin_hours(&self, args: &[Term]) -> Result<Term> {
        let dt = self.datetime_arg(args, "HOURS")?;
        Term::typed_literal(&dt.hour().to_string(), xsd::INTEGER)
    }

    /// `MINUTES(dt)` — the minute component (0-59) of a dateTime.
    fn builtin_minutes(&self, args: &[Term]) -> Result<Term> {
        let dt = self.datetime_arg(args, "MINUTES")?;
        Term::typed_literal(&dt.minute().to_string(), xsd::INTEGER)
    }

    /// `SECONDS(dt)` — the seconds component (0-59) of a dateTime as an
    /// xsd:decimal.
    fn builtin_seconds(&self, args: &[Term]) -> Result<Term> {
        let dt = self.datetime_arg(args, "SECONDS")?;
        Term::typed_literal(&dt.second().to_string(), xsd::DECIMAL)
    }

    /// `TIMEZONE(dt)` — the timezone of a dateTime as an xsd:dayTimeDuration.
    /// A dateTime with no timezone is an error (per SPARQL).
    fn builtin_timezone(&self, args: &[Term]) -> Result<Term> {
        let offset = self.datetime_offset_arg(args, "TIMEZONE")?;
        let total_secs = offset.local_minus_utc();
        let sign = if total_secs < 0 { "-" } else { "" };
        let abs = total_secs.unsigned_abs();
        let hours = abs / 3600;
        let minutes = (abs % 3600) / 60;
        let seconds = abs % 60;
        let mut duration = format!("{sign}PT");
        if hours > 0 {
            duration.push_str(&format!("{hours}H"));
        }
        if minutes > 0 {
            duration.push_str(&format!("{minutes}M"));
        }
        if seconds > 0 || (hours == 0 && minutes == 0) {
            duration.push_str(&format!("{seconds}S"));
        }
        Term::typed_literal(
            &duration,
            "http://www.w3.org/2001/XMLSchema#dayTimeDuration",
        )
    }

    /// `TZ(dt)` — the timezone of a dateTime as a simple literal (`"Z"`, `"+05:00"`,
    /// or the empty string when absent).
    fn builtin_tz(&self, args: &[Term]) -> Result<Term> {
        if args.len() != 1 {
            bail!("TZ expects 1 argument");
        }
        match &args[0] {
            Term::Literal(lit) => {
                if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(&lit.lexical_form) {
                    let offset = dt.offset().local_minus_utc();
                    if offset == 0 {
                        Ok(Term::literal("Z"))
                    } else {
                        let sign = if offset < 0 { '-' } else { '+' };
                        let abs = offset.unsigned_abs();
                        Ok(Term::literal(&format!(
                            "{sign}{:02}:{:02}",
                            abs / 3600,
                            (abs % 3600) / 60
                        )))
                    }
                } else {
                    // A dateTime with no timezone yields the empty string.
                    Ok(Term::literal(""))
                }
            }
            _ => bail!("TZ requires a dateTime literal argument"),
        }
    }

    /// `UUID()` — a fresh `urn:uuid:` IRI.
    fn builtin_uuid(&self, args: &[Term]) -> Result<Term> {
        if !args.is_empty() {
            bail!("UUID expects no arguments");
        }
        Ok(Term::iri(&format!("urn:uuid:{}", uuid::Uuid::new_v4())))
    }

    /// `STRUUID()` — a fresh UUID as a simple string literal.
    fn builtin_struuid(&self, args: &[Term]) -> Result<Term> {
        if !args.is_empty() {
            bail!("STRUUID expects no arguments");
        }
        Ok(Term::literal(&uuid::Uuid::new_v4().to_string()))
    }

    /// `MD5(str)` — lowercase hex MD5 digest.
    fn builtin_md5(&self, args: &[Term]) -> Result<Term> {
        let input = self.hash_input(args, "MD5")?;
        Ok(Term::literal(&format!(
            "{:x}",
            md5::compute(input.as_bytes())
        )))
    }

    /// `SHA1(str)` — lowercase hex SHA-1 digest.
    fn builtin_sha1(&self, args: &[Term]) -> Result<Term> {
        use sha1::{Digest, Sha1};
        let input = self.hash_input(args, "SHA1")?;
        let mut hasher = Sha1::new();
        hasher.update(input.as_bytes());
        Ok(Term::literal(&hex::encode(hasher.finalize())))
    }

    /// `SHA256(str)` — lowercase hex SHA-256 digest.
    fn builtin_sha256(&self, args: &[Term]) -> Result<Term> {
        use sha2::{Digest, Sha256};
        let input = self.hash_input(args, "SHA256")?;
        let mut hasher = Sha256::new();
        hasher.update(input.as_bytes());
        Ok(Term::literal(&hex::encode(hasher.finalize())))
    }

    /// `SHA384(str)` — lowercase hex SHA-384 digest.
    fn builtin_sha384(&self, args: &[Term]) -> Result<Term> {
        use sha2::{Digest, Sha384};
        let input = self.hash_input(args, "SHA384")?;
        let mut hasher = Sha384::new();
        hasher.update(input.as_bytes());
        Ok(Term::literal(&hex::encode(hasher.finalize())))
    }

    /// `SHA512(str)` — lowercase hex SHA-512 digest.
    fn builtin_sha512(&self, args: &[Term]) -> Result<Term> {
        use sha2::{Digest, Sha512};
        let input = self.hash_input(args, "SHA512")?;
        let mut hasher = Sha512::new();
        hasher.update(input.as_bytes());
        Ok(Term::literal(&hex::encode(hasher.finalize())))
    }

    /// Extract the string argument for a single-argument hash function.
    fn hash_input(&self, args: &[Term], name: &str) -> Result<String> {
        if args.len() != 1 {
            bail!("{name} expects 1 argument");
        }
        match &args[0] {
            Term::Literal(lit) => Ok(lit.lexical_form.clone()),
            _ => bail!("{name} requires a string literal argument"),
        }
    }

    /// Get the lexical value of a term as a plain `String`, for functions that
    /// operate on the string form (IRIs, literals, blank nodes).
    fn string_value(&self, term: &Term) -> Result<String> {
        match self.builtin_str(std::slice::from_ref(term))? {
            Term::Literal(lit) => Ok(lit.lexical_form),
            _ => bail!("expected a string value"),
        }
    }

    /// Get a literal's lexical form and language tag (for functions that must
    /// preserve `arg1`'s language on a substring result).
    fn literal_parts(&self, term: &Term) -> Result<(String, Option<String>)> {
        match term {
            Term::Literal(lit) => Ok((lit.lexical_form.clone(), lit.language_tag.clone())),
            other => Ok((self.string_value(other)?, None)),
        }
    }

    /// Build a string-valued result literal, carrying `lang` when present.
    fn make_str_literal(&self, value: &str, lang: &Option<String>) -> Term {
        match lang {
            Some(l) if !value.is_empty() => Term::lang_literal(value, l),
            _ => Term::literal(value),
        }
    }

    /// Parse the single dateTime argument of a time-component accessor into a
    /// timezone-aware `DateTime`, accepting both offset and naive forms.
    fn datetime_arg(
        &self,
        args: &[Term],
        name: &str,
    ) -> Result<chrono::DateTime<chrono::FixedOffset>> {
        if args.len() != 1 {
            bail!("{name} expects 1 argument");
        }
        match &args[0] {
            Term::Literal(lit) => {
                if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(&lit.lexical_form) {
                    Ok(dt)
                } else if let Ok(naive) =
                    chrono::NaiveDateTime::parse_from_str(&lit.lexical_form, "%Y-%m-%dT%H:%M:%S")
                {
                    let offset = chrono::FixedOffset::east_opt(0)
                        .ok_or_else(|| anyhow!("invalid UTC offset"))?;
                    naive
                        .and_local_timezone(offset)
                        .single()
                        .ok_or_else(|| anyhow!("ambiguous local dateTime"))
                } else {
                    bail!("{name} requires a valid dateTime literal")
                }
            }
            _ => bail!("{name} requires a dateTime literal argument"),
        }
    }

    /// Parse the dateTime argument of TIMEZONE, requiring an explicit timezone.
    fn datetime_offset_arg(&self, args: &[Term], name: &str) -> Result<chrono::FixedOffset> {
        if args.len() != 1 {
            bail!("{name} expects 1 argument");
        }
        match &args[0] {
            Term::Literal(lit) => {
                let dt = chrono::DateTime::parse_from_rfc3339(&lit.lexical_form)
                    .map_err(|_| anyhow!("{name} requires a dateTime with a timezone"))?;
                Ok(*dt.offset())
            }
            _ => bail!("{name} requires a dateTime literal argument"),
        }
    }

    /// Helper to create boolean term
    fn bool_term(&self, value: bool) -> Term {
        Term::typed_literal(if value { "true" } else { "false" }, xsd::BOOLEAN)
            .expect("boolean literal should always be valid")
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

/// Convert an evaluated [`Term`] into an extension-system [`Value`] so a
/// registered custom function can be invoked. Typed numeric/boolean literals
/// are lowered to the corresponding scalar `Value`; everything else is carried
/// losslessly as `Value::Literal` / `Value::Iri` / `Value::BlankNode`.
fn term_to_extension_value(term: &Term) -> Result<crate::extensions::Value> {
    use crate::extensions::Value;
    match term {
        Term::Iri(iri) => Ok(Value::Iri(iri.clone())),
        Term::BlankNode(id) => Ok(Value::BlankNode(id.clone())),
        Term::Literal(lit) => {
            if lit.language_tag.is_some() {
                return Ok(Value::Literal {
                    value: lit.lexical_form.clone(),
                    language: lit.language_tag.clone(),
                    datatype: Some(lit.datatype.clone()),
                });
            }
            match lit.datatype.as_str() {
                xsd::INTEGER | xsd::LONG | xsd::INT | xsd::SHORT | xsd::BYTE => lit
                    .lexical_form
                    .parse::<i64>()
                    .map(Value::Integer)
                    .map_err(|_| anyhow!("invalid integer literal: {}", lit.lexical_form)),
                xsd::DOUBLE | xsd::FLOAT | xsd::DECIMAL => lit
                    .lexical_form
                    .parse::<f64>()
                    .map(Value::Float)
                    .map_err(|_| anyhow!("invalid numeric literal: {}", lit.lexical_form)),
                xsd::BOOLEAN => match lit.lexical_form.as_str() {
                    "true" | "1" => Ok(Value::Boolean(true)),
                    "false" | "0" => Ok(Value::Boolean(false)),
                    other => Err(anyhow!("invalid boolean literal: {other}")),
                },
                xsd::STRING => Ok(Value::String(lit.lexical_form.clone())),
                _ => Ok(Value::Literal {
                    value: lit.lexical_form.clone(),
                    language: None,
                    datatype: Some(lit.datatype.clone()),
                }),
            }
        }
        other => bail!("cannot pass term {other:?} to an extension function"),
    }
}

/// Convert an extension-system [`Value`] returned by a custom function back into
/// a [`Term`]. An unbound (`Null`) or non-representable value is an explicit
/// error rather than a silent empty result.
fn extension_value_to_term(value: &crate::extensions::Value) -> Result<Term> {
    use crate::extensions::Value;
    match value {
        Value::String(s) => Ok(Term::literal(s)),
        Value::Integer(i) => Term::typed_literal(&i.to_string(), xsd::INTEGER),
        Value::Float(f) => Term::typed_literal(&f.to_string(), xsd::DOUBLE),
        Value::Boolean(b) => Term::typed_literal(&b.to_string(), xsd::BOOLEAN),
        Value::DateTime(dt) => Term::typed_literal(&dt.to_rfc3339(), xsd::DATE_TIME),
        Value::Iri(iri) => Ok(Term::iri(iri)),
        Value::BlankNode(id) => Ok(Term::blank_node(id)),
        Value::Literal {
            value,
            language,
            datatype,
        } => match (language, datatype) {
            (Some(lang), _) => Ok(Term::lang_literal(value, lang)),
            (None, Some(dt)) => Term::typed_literal(value, dt),
            (None, None) => Ok(Term::literal(value)),
        },
        Value::Null => bail!("extension function returned an unbound (null) value"),
        other => bail!("unsupported extension function result: {other:?}"),
    }
}

impl ExpressionEvaluator {
    /// Test-only shim exposing `evaluate_function` with already-evaluated term
    /// arguments (wrapping them as literal/IRI expressions).
    #[cfg(test)]
    fn evaluate_function_test(&self, name: &str, args: &[Term]) -> Result<Term> {
        let exprs: Vec<AlgebraExpression> = args
            .iter()
            .map(|t| match t {
                Term::Iri(iri) => {
                    AlgebraExpression::Iri(oxirs_core::model::NamedNode::new_unchecked(iri.clone()))
                }
                Term::Literal(lit) => AlgebraExpression::Literal(crate::algebra::Literal {
                    value: lit.lexical_form.clone(),
                    language: lit.language_tag.clone(),
                    datatype: Some(oxirs_core::model::NamedNode::new_unchecked(
                        lit.datatype.clone(),
                    )),
                }),
                Term::BlankNode(id) => AlgebraExpression::Literal(crate::algebra::Literal {
                    value: id.clone(),
                    language: None,
                    datatype: None,
                }),
                _ => AlgebraExpression::Literal(crate::algebra::Literal {
                    value: String::new(),
                    language: None,
                    datatype: None,
                }),
            })
            .collect();
        self.evaluate_function(name, &exprs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxirs_core::{NamedNode, Variable};

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

    fn eval_fn(name: &str, args: &[Term]) -> Result<Term> {
        let evaluator = ExpressionEvaluator::new(Arc::new(ExtensionRegistry::new()));
        evaluator.evaluate_function_test(name, args)
    }

    fn bool_lit(value: bool) -> Term {
        Term::typed_literal(if value { "true" } else { "false" }, xsd::BOOLEAN).unwrap()
    }

    /// SHA/MD5 hashes must be real digests matching known vectors.
    #[test]
    fn regression_hash_functions_real() {
        let abc = [Term::literal("abc")];
        assert_eq!(
            eval_fn("md5", &abc).unwrap(),
            Term::literal("900150983cd24fb0d6963f7d28e17f72")
        );
        assert_eq!(
            eval_fn("sha1", &abc).unwrap(),
            Term::literal("a9993e364706816aba3e25717850c26c9cd0d89d")
        );
        assert_eq!(
            eval_fn("sha256", &abc).unwrap(),
            Term::literal("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        );
        assert_eq!(
            eval_fn("sha512", &abc).unwrap(),
            Term::literal(
                "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a\
                 2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f"
            )
        );
    }

    /// REGEX must be a real regular-expression match, not a literal contains.
    #[test]
    fn regression_regex_real() {
        let args = [Term::literal("abc123"), Term::literal("[0-9]+")];
        assert_eq!(eval_fn("regex", &args).unwrap(), bool_lit(true));
        let no = [Term::literal("abc"), Term::literal("^[0-9]+$")];
        assert_eq!(eval_fn("regex", &no).unwrap(), bool_lit(false));
        // An invalid pattern must fail loudly, never silently return false.
        let bad = [Term::literal("x"), Term::literal("(")];
        assert!(eval_fn("regex", &bad).is_err());
    }

    /// REPLACE must apply the pattern as a regex.
    #[test]
    fn regression_replace_regex() {
        let args = [
            Term::literal("abc123"),
            Term::literal("[0-9]+"),
            Term::literal("#"),
        ];
        assert_eq!(eval_fn("replace", &args).unwrap(), Term::literal("abc#"));
    }

    /// LANG / DATATYPE accessors on literals.
    #[test]
    fn regression_lang_and_datatype() {
        let lang_lit = [Term::lang_literal("chat", "en")];
        assert_eq!(eval_fn("lang", &lang_lit).unwrap(), Term::literal("en"));
        let plain = [Term::literal("hi")];
        assert_eq!(eval_fn("lang", &plain).unwrap(), Term::literal(""));
        let typed = [Term::typed_literal("42", xsd::INTEGER).unwrap()];
        assert_eq!(
            eval_fn("datatype", &typed).unwrap(),
            Term::iri(xsd::INTEGER)
        );
    }

    /// STRBEFORE / STRAFTER / ENCODE_FOR_URI / sameTerm / langMatches.
    #[test]
    fn regression_string_and_term_builtins() {
        assert_eq!(
            eval_fn("strbefore", &[Term::literal("abc"), Term::literal("b")]).unwrap(),
            Term::literal("a")
        );
        assert_eq!(
            eval_fn("strafter", &[Term::literal("abc"), Term::literal("b")]).unwrap(),
            Term::literal("c")
        );
        assert_eq!(
            eval_fn("encode_for_uri", &[Term::literal("a b/c")]).unwrap(),
            Term::literal("a%20b%2Fc")
        );
        assert_eq!(
            eval_fn("sameterm", &[Term::iri("http://x"), Term::iri("http://x")]).unwrap(),
            bool_lit(true)
        );
        assert_eq!(
            eval_fn(
                "langmatches",
                &[Term::literal("en-US"), Term::literal("en")]
            )
            .unwrap(),
            bool_lit(true)
        );
    }

    /// Time-component accessors on a dateTime literal.
    #[test]
    fn regression_datetime_components() {
        let dt = [Term::typed_literal("2020-01-02T13:24:35Z", xsd::DATE_TIME).unwrap()];
        assert_eq!(
            eval_fn("hours", &dt).unwrap(),
            Term::typed_literal("13", xsd::INTEGER).unwrap()
        );
        assert_eq!(
            eval_fn("minutes", &dt).unwrap(),
            Term::typed_literal("24", xsd::INTEGER).unwrap()
        );
    }

    /// An unknown function must fail loudly, never silently return a value.
    #[test]
    fn regression_unknown_function_fails_loud() {
        assert!(eval_fn("http://example.org/nope", &[Term::literal("x")]).is_err());
    }
}
