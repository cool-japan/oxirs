//! Evaluation environment and evaluator for SHACL expressions.
//!
//! Provides [`ExpressionContext`] — the environment a [`ShaclExpression`] is
//! evaluated against — and [`ExpressionEvaluator`], a stateless evaluator that
//! recursively reduces an expression AST to a [`ShaclValue`].
//!
//! Reference: <https://www.w3.org/TR/shacl-af/#ExpressionConstraintComponent>

use std::collections::HashMap;

use crate::{Result, ShaclError};

use super::expression_constraint_types::{ShaclExpression, ShaclPath, ShaclValue};

// ---------------------------------------------------------------------------
// ExpressionContext — evaluation environment
// ---------------------------------------------------------------------------

/// The environment in which a SHACL expression is evaluated.
#[derive(Debug, Clone)]
pub struct ExpressionContext {
    /// The current focus node IRI
    pub this_node: String,
    /// The current value node (for property shape expressions)
    pub value_node: Option<String>,
    /// Additional variable bindings
    pub bindings: HashMap<String, ShaclValue>,
    /// Path resolver: given (focus_node, path) returns a list of values
    pub path_resolver: Option<PathResolver>,
}

/// A function pointer type for resolving property path values from the graph.
pub type PathResolver = fn(&str, &ShaclPath) -> Vec<ShaclValue>;

impl ExpressionContext {
    /// Create a minimal context with just a focus node.
    pub fn new(this_node: impl Into<String>) -> Self {
        Self {
            this_node: this_node.into(),
            value_node: None,
            bindings: HashMap::new(),
            path_resolver: None,
        }
    }

    /// Set the value node (builder pattern).
    pub fn with_value(mut self, value: impl Into<String>) -> Self {
        self.value_node = Some(value.into());
        self
    }

    /// Add a variable binding (builder pattern).
    pub fn bind(mut self, var: impl Into<String>, val: ShaclValue) -> Self {
        self.bindings.insert(var.into(), val);
        self
    }

    /// Set the path resolver (builder pattern).
    pub fn with_path_resolver(mut self, resolver: PathResolver) -> Self {
        self.path_resolver = Some(resolver);
        self
    }
}

// ---------------------------------------------------------------------------
// ExpressionEvaluator
// ---------------------------------------------------------------------------

/// Stateless evaluator for SHACL expressions.
pub struct ExpressionEvaluator;

impl ExpressionEvaluator {
    /// Evaluate a SHACL expression within the given context.
    pub fn evaluate(expr: &ShaclExpression, ctx: &ExpressionContext) -> Result<ShaclValue> {
        match expr {
            // ---- Primitives ------------------------------------------------
            ShaclExpression::Literal(val) => Ok(val.clone()),

            ShaclExpression::Variable(name) => {
                let resolved = match name.as_str() {
                    "this" | "$this" => ShaclValue::Iri(ctx.this_node.clone()),
                    "value" | "$value" => ctx
                        .value_node
                        .as_ref()
                        .map(|v| ShaclValue::Iri(v.clone()))
                        .unwrap_or(ShaclValue::Null),
                    other => ctx.bindings.get(other).cloned().unwrap_or(ShaclValue::Null),
                };
                Ok(resolved)
            }

            ShaclExpression::Path(path) => {
                let values = Self::resolve_path(path, ctx)?;
                // Return the first value or Null
                Ok(values.into_iter().next().unwrap_or(ShaclValue::Null))
            }

            // ---- Arithmetic ------------------------------------------------
            ShaclExpression::Add(l, r) => Self::arithmetic(l, r, ctx, "+"),
            ShaclExpression::Sub(l, r) => Self::arithmetic(l, r, ctx, "-"),
            ShaclExpression::Mul(l, r) => Self::arithmetic(l, r, ctx, "*"),
            ShaclExpression::Div(l, r) => Self::arithmetic(l, r, ctx, "/"),

            // ---- Comparison ------------------------------------------------
            ShaclExpression::Eq(l, r) => {
                let lv = Self::evaluate(l, ctx)?;
                let rv = Self::evaluate(r, ctx)?;
                Ok(ShaclValue::Boolean(lv == rv))
            }
            ShaclExpression::Ne(l, r) => {
                let lv = Self::evaluate(l, ctx)?;
                let rv = Self::evaluate(r, ctx)?;
                Ok(ShaclValue::Boolean(lv != rv))
            }
            ShaclExpression::Lt(l, r) => Self::compare(l, r, ctx, "<"),
            ShaclExpression::Gt(l, r) => Self::compare(l, r, ctx, ">"),
            ShaclExpression::Lte(l, r) => Self::compare(l, r, ctx, "<="),
            ShaclExpression::Gte(l, r) => Self::compare(l, r, ctx, ">="),

            // ---- Logical ---------------------------------------------------
            ShaclExpression::And(l, r) => {
                let lv = Self::evaluate(l, ctx)?;
                if !lv.is_truthy() {
                    return Ok(ShaclValue::Boolean(false));
                }
                let rv = Self::evaluate(r, ctx)?;
                Ok(ShaclValue::Boolean(rv.is_truthy()))
            }
            ShaclExpression::Or(l, r) => {
                let lv = Self::evaluate(l, ctx)?;
                if lv.is_truthy() {
                    return Ok(ShaclValue::Boolean(true));
                }
                let rv = Self::evaluate(r, ctx)?;
                Ok(ShaclValue::Boolean(rv.is_truthy()))
            }
            ShaclExpression::Not(inner) => {
                let iv = Self::evaluate(inner, ctx)?;
                Ok(ShaclValue::Boolean(!iv.is_truthy()))
            }

            // ---- String functions ------------------------------------------
            ShaclExpression::Concat(parts) => {
                let mut result = String::new();
                for part in parts {
                    let v = Self::evaluate(part, ctx)?;
                    result.push_str(&v.as_string());
                }
                Ok(ShaclValue::Literal {
                    value: result,
                    datatype: Some("http://www.w3.org/2001/XMLSchema#string".to_string()),
                    lang: None,
                })
            }
            ShaclExpression::StrLen(inner) => {
                let v = Self::evaluate(inner, ctx)?;
                let len = v.as_string().chars().count() as i64;
                Ok(ShaclValue::Integer(len))
            }
            ShaclExpression::Regex(inner, pattern) => {
                let v = Self::evaluate(inner, ctx)?;
                let re = regex::Regex::new(pattern).map_err(|e| {
                    ShaclError::ConstraintValidation(format!("Invalid regex '{pattern}': {e}"))
                })?;
                Ok(ShaclValue::Boolean(re.is_match(&v.as_string())))
            }
            ShaclExpression::UpperCase(inner) => {
                let v = Self::evaluate(inner, ctx)?;
                Ok(ShaclValue::Literal {
                    value: v.as_string().to_uppercase(),
                    datatype: Some("http://www.w3.org/2001/XMLSchema#string".to_string()),
                    lang: None,
                })
            }
            ShaclExpression::LowerCase(inner) => {
                let v = Self::evaluate(inner, ctx)?;
                Ok(ShaclValue::Literal {
                    value: v.as_string().to_lowercase(),
                    datatype: Some("http://www.w3.org/2001/XMLSchema#string".to_string()),
                    lang: None,
                })
            }

            // ---- Numeric functions -----------------------------------------
            ShaclExpression::Abs(inner) => {
                let v = Self::evaluate(inner, ctx)?;
                match v {
                    ShaclValue::Integer(n) => Ok(ShaclValue::Integer(n.abs())),
                    ShaclValue::Float(x) => Ok(ShaclValue::Float(x.abs())),
                    other => Err(ShaclError::ConstraintValidation(format!(
                        "ABS requires a numeric value, got: {other}"
                    ))),
                }
            }
            ShaclExpression::Floor(inner) => {
                let v = Self::evaluate(inner, ctx)?;
                match v.as_f64() {
                    Some(x) => Ok(ShaclValue::Integer(x.floor() as i64)),
                    None => Err(ShaclError::ConstraintValidation(format!(
                        "FLOOR requires a numeric value, got: {v}"
                    ))),
                }
            }
            ShaclExpression::Ceil(inner) => {
                let v = Self::evaluate(inner, ctx)?;
                match v.as_f64() {
                    Some(x) => Ok(ShaclValue::Integer(x.ceil() as i64)),
                    None => Err(ShaclError::ConstraintValidation(format!(
                        "CEIL requires a numeric value, got: {v}"
                    ))),
                }
            }
            ShaclExpression::Round(inner) => {
                let v = Self::evaluate(inner, ctx)?;
                match v.as_f64() {
                    Some(x) => Ok(ShaclValue::Integer(x.round() as i64)),
                    None => Err(ShaclError::ConstraintValidation(format!(
                        "ROUND requires a numeric value, got: {v}"
                    ))),
                }
            }

            // ---- Aggregate / graph -----------------------------------------
            ShaclExpression::Count(path) => {
                let values = Self::resolve_path(path, ctx)?;
                Ok(ShaclValue::Integer(values.len() as i64))
            }

            // ---- Type functions --------------------------------------------
            ShaclExpression::IsIri(inner) => {
                let v = Self::evaluate(inner, ctx)?;
                Ok(ShaclValue::Boolean(matches!(v, ShaclValue::Iri(_))))
            }
            ShaclExpression::IsLiteral(inner) => {
                let v = Self::evaluate(inner, ctx)?;
                Ok(ShaclValue::Boolean(matches!(v, ShaclValue::Literal { .. })))
            }
            ShaclExpression::Datatype(inner) => {
                let v = Self::evaluate(inner, ctx)?;
                match v {
                    ShaclValue::Literal {
                        datatype: Some(dt), ..
                    } => Ok(ShaclValue::Iri(dt)),
                    ShaclValue::Integer(_) => Ok(ShaclValue::Iri(
                        "http://www.w3.org/2001/XMLSchema#integer".to_string(),
                    )),
                    ShaclValue::Float(_) => Ok(ShaclValue::Iri(
                        "http://www.w3.org/2001/XMLSchema#double".to_string(),
                    )),
                    ShaclValue::Boolean(_) => Ok(ShaclValue::Iri(
                        "http://www.w3.org/2001/XMLSchema#boolean".to_string(),
                    )),
                    _ => Ok(ShaclValue::Null),
                }
            }
            ShaclExpression::Lang(inner) => {
                let v = Self::evaluate(inner, ctx)?;
                match v {
                    ShaclValue::Literal { lang: Some(l), .. } => Ok(ShaclValue::Literal {
                        value: l,
                        datatype: None,
                        lang: None,
                    }),
                    _ => Ok(ShaclValue::Literal {
                        value: String::new(),
                        datatype: None,
                        lang: None,
                    }),
                }
            }
        }
    }

    // ---- Helper: arithmetic -----------------------------------------------

    fn arithmetic(
        l: &ShaclExpression,
        r: &ShaclExpression,
        ctx: &ExpressionContext,
        op: &str,
    ) -> Result<ShaclValue> {
        let lv = Self::evaluate(l, ctx)?;
        let rv = Self::evaluate(r, ctx)?;

        // Try integer arithmetic first (to preserve exact types)
        if let (Some(li), Some(ri)) = (lv.as_i64(), rv.as_i64()) {
            match op {
                "+" => return Ok(ShaclValue::Integer(li + ri)),
                "-" => return Ok(ShaclValue::Integer(li - ri)),
                "*" => return Ok(ShaclValue::Integer(li * ri)),
                "/" => {
                    if ri == 0 {
                        return Err(ShaclError::ConstraintValidation(
                            "Division by zero".to_string(),
                        ));
                    }
                    // Integer division produces a float for semantic correctness
                    return Ok(ShaclValue::Float(li as f64 / ri as f64));
                }
                _ => {}
            }
        }

        // Fall back to floating-point arithmetic
        let lf = lv.as_f64().ok_or_else(|| {
            ShaclError::ConstraintValidation(format!(
                "Arithmetic operator '{op}' requires numeric operands, got: {lv}"
            ))
        })?;
        let rf = rv.as_f64().ok_or_else(|| {
            ShaclError::ConstraintValidation(format!(
                "Arithmetic operator '{op}' requires numeric operands, got: {rv}"
            ))
        })?;

        let result = match op {
            "+" => lf + rf,
            "-" => lf - rf,
            "*" => lf * rf,
            "/" => {
                if rf == 0.0 {
                    return Err(ShaclError::ConstraintValidation(
                        "Division by zero".to_string(),
                    ));
                }
                lf / rf
            }
            _ => unreachable!("unknown arithmetic op: {op}"),
        };

        Ok(ShaclValue::Float(result))
    }

    // ---- Helper: ordered comparison ---------------------------------------

    fn compare(
        l: &ShaclExpression,
        r: &ShaclExpression,
        ctx: &ExpressionContext,
        op: &str,
    ) -> Result<ShaclValue> {
        let lv = Self::evaluate(l, ctx)?;
        let rv = Self::evaluate(r, ctx)?;

        // Numeric comparison
        if let (Some(lf), Some(rf)) = (lv.as_f64(), rv.as_f64()) {
            let result = match op {
                "<" => lf < rf,
                ">" => lf > rf,
                "<=" => lf <= rf,
                ">=" => lf >= rf,
                _ => unreachable!(),
            };
            return Ok(ShaclValue::Boolean(result));
        }

        // String comparison (lexicographic)
        let ls = lv.as_string();
        let rs = rv.as_string();
        let result = match op {
            "<" => ls < rs,
            ">" => ls > rs,
            "<=" => ls <= rs,
            ">=" => ls >= rs,
            _ => unreachable!(),
        };
        Ok(ShaclValue::Boolean(result))
    }

    // ---- Helper: path resolution ------------------------------------------

    fn resolve_path(path: &ShaclPath, ctx: &ExpressionContext) -> Result<Vec<ShaclValue>> {
        match ctx.path_resolver {
            Some(resolver) => Ok(resolver(&ctx.this_node, path)),
            None => {
                // No resolver supplied — return empty (path not evaluable)
                Ok(Vec::new())
            }
        }
    }
}
