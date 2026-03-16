//! SHACL Expression Constraints (sh:ExpressionConstraintComponent)
//!
//! Implements the SHACL Advanced Features expression mechanism, which allows
//! evaluating mathematical and string expressions over RDF node values.
//!
//! Reference: <https://www.w3.org/TR/shacl-af/#ExpressionConstraintComponent>

use std::collections::HashMap;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::{Result, ShaclError};

// ---------------------------------------------------------------------------
// ShaclPath — lightweight inline definition (mirrors the main paths module
// for expression evaluation purposes without creating a circular dependency)
// ---------------------------------------------------------------------------

/// Simplified property path for use within SHACL expressions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ShaclPath {
    /// A direct predicate IRI
    Predicate(String),
    /// Inverse path (`^p`)
    Inverse(Box<ShaclPath>),
    /// Sequence path (`p1 / p2 / ...`)
    Sequence(Vec<ShaclPath>),
    /// Alternative path (`p1 | p2 | ...`)
    Alternative(Vec<ShaclPath>),
    /// Zero-or-more path (`p*`)
    ZeroOrMore(Box<ShaclPath>),
    /// One-or-more path (`p+`)
    OneOrMore(Box<ShaclPath>),
}

impl fmt::Display for ShaclPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ShaclPath::Predicate(iri) => write!(f, "<{iri}>"),
            ShaclPath::Inverse(p) => write!(f, "^{p}"),
            ShaclPath::Sequence(steps) => {
                let parts: Vec<_> = steps.iter().map(|s| s.to_string()).collect();
                write!(f, "{}", parts.join(" / "))
            }
            ShaclPath::Alternative(alts) => {
                let parts: Vec<_> = alts.iter().map(|s| s.to_string()).collect();
                write!(f, "{}", parts.join(" | "))
            }
            ShaclPath::ZeroOrMore(p) => write!(f, "{p}*"),
            ShaclPath::OneOrMore(p) => write!(f, "{p}+"),
        }
    }
}

// ---------------------------------------------------------------------------
// ShaclValue — runtime value type
// ---------------------------------------------------------------------------

/// A typed value produced by evaluating a SHACL expression.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ShaclValue {
    /// An IRI resource
    Iri(String),
    /// An RDF literal with optional datatype and language tag
    Literal {
        value: String,
        datatype: Option<String>,
        lang: Option<String>,
    },
    /// A plain integer (xsd:integer)
    Integer(i64),
    /// A floating-point number (xsd:double)
    Float(f64),
    /// A boolean (xsd:boolean)
    Boolean(bool),
    /// The null / absent value
    Null,
}

impl fmt::Display for ShaclValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ShaclValue::Iri(iri) => write!(f, "<{iri}>"),
            ShaclValue::Literal {
                value,
                datatype,
                lang,
            } => match (datatype, lang) {
                (Some(dt), _) => write!(f, "\"{value}\"^^<{dt}>"),
                (_, Some(l)) => write!(f, "\"{value}\"@{l}"),
                _ => write!(f, "\"{value}\""),
            },
            ShaclValue::Integer(n) => write!(f, "{n}"),
            ShaclValue::Float(x) => write!(f, "{x}"),
            ShaclValue::Boolean(b) => write!(f, "{b}"),
            ShaclValue::Null => write!(f, "null"),
        }
    }
}

impl ShaclValue {
    /// Returns `true` for values that are logically "truthy".
    pub fn is_truthy(&self) -> bool {
        match self {
            ShaclValue::Boolean(b) => *b,
            ShaclValue::Null => false,
            ShaclValue::Integer(n) => *n != 0,
            ShaclValue::Float(x) => *x != 0.0 && !x.is_nan(),
            ShaclValue::Literal { value, .. } => !value.is_empty(),
            ShaclValue::Iri(_) => true,
        }
    }

    /// Attempt to interpret the value as an `f64`.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            ShaclValue::Float(x) => Some(*x),
            ShaclValue::Integer(n) => Some(*n as f64),
            ShaclValue::Literal { value, .. } => value.parse::<f64>().ok(),
            _ => None,
        }
    }

    /// Attempt to interpret the value as an `i64`.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            ShaclValue::Integer(n) => Some(*n),
            ShaclValue::Float(x) => Some(*x as i64),
            ShaclValue::Literal { value, .. } => value.parse::<i64>().ok(),
            _ => None,
        }
    }

    /// Represent the value as a plain string.
    pub fn as_string(&self) -> String {
        match self {
            ShaclValue::Iri(iri) => iri.clone(),
            ShaclValue::Literal { value, .. } => value.clone(),
            ShaclValue::Integer(n) => n.to_string(),
            ShaclValue::Float(x) => x.to_string(),
            ShaclValue::Boolean(b) => b.to_string(),
            ShaclValue::Null => String::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// ShaclExpression — AST
// ---------------------------------------------------------------------------

/// An AST node representing a SHACL expression.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ShaclExpression {
    // ---- Primitives --------------------------------------------------------
    /// A constant value
    Literal(ShaclValue),
    /// A variable reference (`$this`, `$value`, or user-defined names)
    Variable(String),
    /// A graph path expression — evaluated by the context's path resolver
    Path(ShaclPath),

    // ---- Arithmetic --------------------------------------------------------
    Add(Box<ShaclExpression>, Box<ShaclExpression>),
    Sub(Box<ShaclExpression>, Box<ShaclExpression>),
    Mul(Box<ShaclExpression>, Box<ShaclExpression>),
    Div(Box<ShaclExpression>, Box<ShaclExpression>),

    // ---- Comparison --------------------------------------------------------
    Eq(Box<ShaclExpression>, Box<ShaclExpression>),
    Ne(Box<ShaclExpression>, Box<ShaclExpression>),
    Lt(Box<ShaclExpression>, Box<ShaclExpression>),
    Gt(Box<ShaclExpression>, Box<ShaclExpression>),
    Lte(Box<ShaclExpression>, Box<ShaclExpression>),
    Gte(Box<ShaclExpression>, Box<ShaclExpression>),

    // ---- Logical -----------------------------------------------------------
    And(Box<ShaclExpression>, Box<ShaclExpression>),
    Or(Box<ShaclExpression>, Box<ShaclExpression>),
    Not(Box<ShaclExpression>),

    // ---- String functions --------------------------------------------------
    /// Concatenate a list of string values
    Concat(Vec<ShaclExpression>),
    /// Length of a string
    StrLen(Box<ShaclExpression>),
    /// Test a string against a regex pattern
    Regex(Box<ShaclExpression>, String),
    /// Convert to uppercase
    UpperCase(Box<ShaclExpression>),
    /// Convert to lowercase
    LowerCase(Box<ShaclExpression>),

    // ---- Numeric functions -------------------------------------------------
    Abs(Box<ShaclExpression>),
    Floor(Box<ShaclExpression>),
    Ceil(Box<ShaclExpression>),
    Round(Box<ShaclExpression>),

    // ---- Aggregate / graph -------------------------------------------------
    /// Count the values reachable via a path from `$this`
    Count(ShaclPath),

    // ---- Type functions ----------------------------------------------------
    IsIri(Box<ShaclExpression>),
    IsLiteral(Box<ShaclExpression>),
    Datatype(Box<ShaclExpression>),
    Lang(Box<ShaclExpression>),
}

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

// ---------------------------------------------------------------------------
// ExpressionConstraintComponent
// ---------------------------------------------------------------------------

/// The SHACL expression constraint component.
///
/// Applies a SHACL expression to each value node. The constraint is violated
/// whenever the expression evaluates to a falsy value.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpressionConstraintComponent {
    /// The expression to evaluate
    pub expression: ShaclExpression,
    /// Optional violation message
    pub message: Option<String>,
    /// Whether this constraint is deactivated
    pub deactivated: bool,
}

impl ExpressionConstraintComponent {
    /// Create a new expression constraint.
    pub fn new(expression: ShaclExpression) -> Self {
        Self {
            expression,
            message: None,
            deactivated: false,
        }
    }

    /// Set a violation message (builder pattern).
    pub fn with_message(mut self, msg: impl Into<String>) -> Self {
        self.message = Some(msg.into());
        self
    }

    /// Deactivate this constraint (builder pattern).
    pub fn deactivated(mut self) -> Self {
        self.deactivated = true;
        self
    }

    /// Evaluate the expression for a focus node and return whether it is valid.
    pub fn evaluate(&self, ctx: &ExpressionContext) -> Result<ExpressionConstraintResult> {
        if self.deactivated {
            return Ok(ExpressionConstraintResult {
                focus_node: ctx.this_node.clone(),
                is_valid: true,
                value: ShaclValue::Null,
                message: None,
            });
        }

        let value = ExpressionEvaluator::evaluate(&self.expression, ctx)?;
        let is_valid = value.is_truthy();

        let message = if is_valid {
            None
        } else {
            Some(
                self.message
                    .clone()
                    .unwrap_or_else(|| format!("Expression constraint failed: {value}")),
            )
        };

        Ok(ExpressionConstraintResult {
            focus_node: ctx.this_node.clone(),
            is_valid,
            value,
            message,
        })
    }
}

/// Result of evaluating an `ExpressionConstraintComponent`.
#[derive(Debug, Clone)]
pub struct ExpressionConstraintResult {
    /// The focus node that was validated
    pub focus_node: String,
    /// Whether the expression evaluated to a truthy value
    pub is_valid: bool,
    /// The actual evaluated value
    pub value: ShaclValue,
    /// Violation message (None when valid)
    pub message: Option<String>,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx(this: &str) -> ExpressionContext {
        ExpressionContext::new(this)
    }

    // ---- Literal ----------------------------------------------------------

    #[test]
    fn test_literal_integer() {
        let expr = ShaclExpression::Literal(ShaclValue::Integer(42));
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/a"))
            .expect("evaluation should succeed");
        assert_eq!(result, ShaclValue::Integer(42));
    }

    #[test]
    fn test_literal_float() {
        let expr = ShaclExpression::Literal(ShaclValue::Float(std::f64::consts::PI));
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/a"))
            .expect("evaluation should succeed");
        assert_eq!(result, ShaclValue::Float(std::f64::consts::PI));
    }

    // ---- Variables --------------------------------------------------------

    #[test]
    fn test_variable_this() {
        let expr = ShaclExpression::Variable("this".to_string());
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/Alice"))
            .expect("evaluation should succeed");
        assert_eq!(result, ShaclValue::Iri("http://ex/Alice".to_string()));
    }

    #[test]
    fn test_variable_value_absent() {
        let expr = ShaclExpression::Variable("value".to_string());
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/a"))
            .expect("evaluation should succeed");
        assert_eq!(result, ShaclValue::Null);
    }

    #[test]
    fn test_variable_custom_binding() {
        let expr = ShaclExpression::Variable("myVar".to_string());
        let c = ctx("http://ex/a").bind("myVar", ShaclValue::Integer(99));
        let result = ExpressionEvaluator::evaluate(&expr, &c).expect("evaluation should succeed");
        assert_eq!(result, ShaclValue::Integer(99));
    }

    // ---- Arithmetic -------------------------------------------------------

    #[test]
    fn test_add_integers() {
        let expr = ShaclExpression::Add(
            Box::new(ShaclExpression::Literal(ShaclValue::Integer(3))),
            Box::new(ShaclExpression::Literal(ShaclValue::Integer(4))),
        );
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/a"))
            .expect("evaluation should succeed");
        assert_eq!(result, ShaclValue::Integer(7));
    }

    #[test]
    fn test_sub_integers() {
        let expr = ShaclExpression::Sub(
            Box::new(ShaclExpression::Literal(ShaclValue::Integer(10))),
            Box::new(ShaclExpression::Literal(ShaclValue::Integer(3))),
        );
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/a"))
            .expect("evaluation should succeed");
        assert_eq!(result, ShaclValue::Integer(7));
    }

    #[test]
    fn test_mul_integers() {
        let expr = ShaclExpression::Mul(
            Box::new(ShaclExpression::Literal(ShaclValue::Integer(6))),
            Box::new(ShaclExpression::Literal(ShaclValue::Integer(7))),
        );
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/a"))
            .expect("evaluation should succeed");
        assert_eq!(result, ShaclValue::Integer(42));
    }

    #[test]
    fn test_div_integers_produces_float() {
        let expr = ShaclExpression::Div(
            Box::new(ShaclExpression::Literal(ShaclValue::Integer(7))),
            Box::new(ShaclExpression::Literal(ShaclValue::Integer(2))),
        );
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/a"))
            .expect("evaluation should succeed");
        assert_eq!(result, ShaclValue::Float(3.5));
    }

    #[test]
    fn test_div_by_zero() {
        let expr = ShaclExpression::Div(
            Box::new(ShaclExpression::Literal(ShaclValue::Integer(1))),
            Box::new(ShaclExpression::Literal(ShaclValue::Integer(0))),
        );
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/a"));
        assert!(result.is_err());
    }

    // ---- Comparison -------------------------------------------------------

    #[test]
    fn test_lt_true() {
        let expr = ShaclExpression::Lt(
            Box::new(ShaclExpression::Literal(ShaclValue::Integer(3))),
            Box::new(ShaclExpression::Literal(ShaclValue::Integer(5))),
        );
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/a"))
            .expect("evaluation should succeed");
        assert_eq!(result, ShaclValue::Boolean(true));
    }

    #[test]
    fn test_gte_false() {
        let expr = ShaclExpression::Gte(
            Box::new(ShaclExpression::Literal(ShaclValue::Integer(3))),
            Box::new(ShaclExpression::Literal(ShaclValue::Integer(5))),
        );
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/a"))
            .expect("evaluation should succeed");
        assert_eq!(result, ShaclValue::Boolean(false));
    }

    #[test]
    fn test_eq_same_integer() {
        let expr = ShaclExpression::Eq(
            Box::new(ShaclExpression::Literal(ShaclValue::Integer(7))),
            Box::new(ShaclExpression::Literal(ShaclValue::Integer(7))),
        );
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/a"))
            .expect("evaluation should succeed");
        assert_eq!(result, ShaclValue::Boolean(true));
    }

    // ---- Logical ----------------------------------------------------------

    #[test]
    fn test_and_true() {
        let expr = ShaclExpression::And(
            Box::new(ShaclExpression::Literal(ShaclValue::Boolean(true))),
            Box::new(ShaclExpression::Literal(ShaclValue::Boolean(true))),
        );
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/a"))
            .expect("evaluation should succeed");
        assert_eq!(result, ShaclValue::Boolean(true));
    }

    #[test]
    fn test_and_short_circuit() {
        let expr = ShaclExpression::And(
            Box::new(ShaclExpression::Literal(ShaclValue::Boolean(false))),
            // This would error if evaluated
            Box::new(ShaclExpression::Div(
                Box::new(ShaclExpression::Literal(ShaclValue::Integer(1))),
                Box::new(ShaclExpression::Literal(ShaclValue::Integer(0))),
            )),
        );
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/a"))
            .expect("short-circuit should avoid division by zero");
        assert_eq!(result, ShaclValue::Boolean(false));
    }

    #[test]
    fn test_or_true() {
        let expr = ShaclExpression::Or(
            Box::new(ShaclExpression::Literal(ShaclValue::Boolean(false))),
            Box::new(ShaclExpression::Literal(ShaclValue::Boolean(true))),
        );
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/a"))
            .expect("evaluation should succeed");
        assert_eq!(result, ShaclValue::Boolean(true));
    }

    #[test]
    fn test_not() {
        let expr = ShaclExpression::Not(Box::new(ShaclExpression::Literal(ShaclValue::Boolean(
            false,
        ))));
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/a"))
            .expect("evaluation should succeed");
        assert_eq!(result, ShaclValue::Boolean(true));
    }

    // ---- String functions -------------------------------------------------

    #[test]
    fn test_concat() {
        let expr = ShaclExpression::Concat(vec![
            ShaclExpression::Literal(ShaclValue::Literal {
                value: "Hello".to_string(),
                datatype: None,
                lang: None,
            }),
            ShaclExpression::Literal(ShaclValue::Literal {
                value: ", World".to_string(),
                datatype: None,
                lang: None,
            }),
        ]);
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/a"))
            .expect("evaluation should succeed");
        assert_eq!(result.as_string(), "Hello, World");
    }

    #[test]
    fn test_strlen() {
        let expr =
            ShaclExpression::StrLen(Box::new(ShaclExpression::Literal(ShaclValue::Literal {
                value: "hello".to_string(),
                datatype: None,
                lang: None,
            })));
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/a"))
            .expect("evaluation should succeed");
        assert_eq!(result, ShaclValue::Integer(5));
    }

    #[test]
    fn test_regex_match() {
        let expr = ShaclExpression::Regex(
            Box::new(ShaclExpression::Literal(ShaclValue::Literal {
                value: "hello123".to_string(),
                datatype: None,
                lang: None,
            })),
            r"^\w+\d+$".to_string(),
        );
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/a"))
            .expect("evaluation should succeed");
        assert_eq!(result, ShaclValue::Boolean(true));
    }

    #[test]
    fn test_regex_no_match() {
        let expr = ShaclExpression::Regex(
            Box::new(ShaclExpression::Literal(ShaclValue::Literal {
                value: "hello".to_string(),
                datatype: None,
                lang: None,
            })),
            r"^\d+$".to_string(),
        );
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/a"))
            .expect("evaluation should succeed");
        assert_eq!(result, ShaclValue::Boolean(false));
    }

    #[test]
    fn test_uppercase() {
        let expr =
            ShaclExpression::UpperCase(Box::new(ShaclExpression::Literal(ShaclValue::Literal {
                value: "hello".to_string(),
                datatype: None,
                lang: None,
            })));
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/a"))
            .expect("evaluation should succeed");
        assert_eq!(result.as_string(), "HELLO");
    }

    // ---- Numeric functions ------------------------------------------------

    #[test]
    fn test_abs_negative() {
        let expr =
            ShaclExpression::Abs(Box::new(ShaclExpression::Literal(ShaclValue::Integer(-7))));
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/a"))
            .expect("evaluation should succeed");
        assert_eq!(result, ShaclValue::Integer(7));
    }

    #[test]
    fn test_floor() {
        let expr =
            ShaclExpression::Floor(Box::new(ShaclExpression::Literal(ShaclValue::Float(3.7))));
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/a"))
            .expect("evaluation should succeed");
        assert_eq!(result, ShaclValue::Integer(3));
    }

    #[test]
    fn test_ceil() {
        let expr =
            ShaclExpression::Ceil(Box::new(ShaclExpression::Literal(ShaclValue::Float(3.1))));
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/a"))
            .expect("evaluation should succeed");
        assert_eq!(result, ShaclValue::Integer(4));
    }

    #[test]
    fn test_round() {
        let expr =
            ShaclExpression::Round(Box::new(ShaclExpression::Literal(ShaclValue::Float(2.5))));
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/a"))
            .expect("evaluation should succeed");
        assert_eq!(result, ShaclValue::Integer(3));
    }

    // ---- Type functions ---------------------------------------------------

    #[test]
    fn test_is_iri() {
        let expr = ShaclExpression::IsIri(Box::new(ShaclExpression::Variable("this".to_string())));
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/Alice"))
            .expect("evaluation should succeed");
        assert_eq!(result, ShaclValue::Boolean(true));
    }

    #[test]
    fn test_is_literal() {
        let expr =
            ShaclExpression::IsLiteral(Box::new(ShaclExpression::Literal(ShaclValue::Integer(5))));
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/a"))
            .expect("evaluation should succeed");
        // Integer is not a Literal variant in our type system
        assert_eq!(result, ShaclValue::Boolean(false));
    }

    #[test]
    fn test_datatype_integer() {
        let expr =
            ShaclExpression::Datatype(Box::new(ShaclExpression::Literal(ShaclValue::Integer(42))));
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/a"))
            .expect("evaluation should succeed");
        assert_eq!(
            result,
            ShaclValue::Iri("http://www.w3.org/2001/XMLSchema#integer".to_string())
        );
    }

    // ---- ExpressionConstraintComponent -----------------------------------

    #[test]
    fn test_expression_constraint_valid() {
        let constraint =
            ExpressionConstraintComponent::new(ShaclExpression::Literal(ShaclValue::Boolean(true)));
        let result = constraint
            .evaluate(&ctx("http://ex/Alice"))
            .expect("evaluation should succeed");
        assert!(result.is_valid);
        assert!(result.message.is_none());
    }

    #[test]
    fn test_expression_constraint_invalid() {
        let constraint = ExpressionConstraintComponent::new(ShaclExpression::Literal(
            ShaclValue::Boolean(false),
        ))
        .with_message("Value must be truthy");

        let result = constraint
            .evaluate(&ctx("http://ex/Alice"))
            .expect("evaluation should succeed");
        assert!(!result.is_valid);
        assert_eq!(result.message, Some("Value must be truthy".to_string()));
    }

    #[test]
    fn test_expression_constraint_deactivated() {
        // Even though the expression is always false, deactivated means valid
        let constraint = ExpressionConstraintComponent::new(ShaclExpression::Literal(
            ShaclValue::Boolean(false),
        ))
        .deactivated();

        let result = constraint
            .evaluate(&ctx("http://ex/Alice"))
            .expect("evaluation should succeed");
        assert!(result.is_valid);
    }

    #[test]
    fn test_complex_expression_age_range() {
        // Constraint: age >= 18 && age <= 120
        let age_var = ShaclExpression::Variable("age".to_string());
        let expr = ShaclExpression::And(
            Box::new(ShaclExpression::Gte(
                Box::new(age_var.clone()),
                Box::new(ShaclExpression::Literal(ShaclValue::Integer(18))),
            )),
            Box::new(ShaclExpression::Lte(
                Box::new(age_var),
                Box::new(ShaclExpression::Literal(ShaclValue::Integer(120))),
            )),
        );

        let constraint =
            ExpressionConstraintComponent::new(expr).with_message("Age must be between 18 and 120");

        // Valid age
        let c_valid = ctx("http://ex/Alice").bind("age", ShaclValue::Integer(30));
        let r = constraint
            .evaluate(&c_valid)
            .expect("evaluation should succeed");
        assert!(r.is_valid);

        // Invalid age (too young)
        let c_invalid = ctx("http://ex/Bob").bind("age", ShaclValue::Integer(15));
        let r2 = constraint
            .evaluate(&c_invalid)
            .expect("evaluation should succeed");
        assert!(!r2.is_valid);
        assert!(r2.message.is_some());
    }
}

// ---------------------------------------------------------------------------
// Extended expression constraint tests — using only actual enum variants
// ---------------------------------------------------------------------------

#[cfg(test)]
mod extended_expression_tests {
    use super::*;

    fn ctx(this: &str) -> ExpressionContext {
        ExpressionContext::new(this)
    }

    fn lit_str(s: &str) -> ShaclExpression {
        ShaclExpression::Literal(ShaclValue::Literal {
            value: s.to_string(),
            datatype: Some("http://www.w3.org/2001/XMLSchema#string".to_string()),
            lang: None,
        })
    }

    fn lit_int(n: i64) -> ShaclExpression {
        ShaclExpression::Literal(ShaclValue::Integer(n))
    }

    fn lit_bool(b: bool) -> ShaclExpression {
        ShaclExpression::Literal(ShaclValue::Boolean(b))
    }

    // ---- ExpressionContext fields ---------------------------------------

    #[test]
    fn test_expression_context_this_node_field() {
        let c = ctx("http://ex/Alice");
        assert_eq!(c.this_node, "http://ex/Alice");
    }

    #[test]
    fn test_expression_context_bind_and_lookup() {
        let c = ctx("http://ex/X").bind("score", ShaclValue::Integer(100));
        let expr = ShaclExpression::Variable("score".to_string());
        let v = ExpressionEvaluator::evaluate(&expr, &c).expect("evaluate");
        assert_eq!(v, ShaclValue::Integer(100));
    }

    #[test]
    fn test_expression_context_overwrite_binding() {
        let c = ctx("http://ex/X")
            .bind("x", ShaclValue::Integer(1))
            .bind("x", ShaclValue::Integer(2));
        let expr = ShaclExpression::Variable("x".to_string());
        let v = ExpressionEvaluator::evaluate(&expr, &c).expect("evaluate");
        assert_eq!(v, ShaclValue::Integer(2));
    }

    #[test]
    fn test_expression_context_value_node_default_none() {
        let c = ctx("http://ex/A");
        assert!(c.value_node.is_none());
    }

    #[test]
    fn test_expression_context_with_value() {
        let c = ctx("http://ex/A").with_value("http://ex/Value1");
        assert_eq!(c.value_node.as_deref(), Some("http://ex/Value1"));
    }

    // ---- Nested arithmetic expressions ---------------------------------

    #[test]
    fn test_nested_add_mul() {
        // (2 + 3) * 4 = 20
        let inner_add = ShaclExpression::Add(Box::new(lit_int(2)), Box::new(lit_int(3)));
        let expr = ShaclExpression::Mul(Box::new(inner_add), Box::new(lit_int(4)));
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/A")).expect("evaluate");
        assert_eq!(result, ShaclValue::Integer(20));
    }

    #[test]
    fn test_nested_sub_div() {
        // (10 - 4) / 2 = 3
        let inner_sub = ShaclExpression::Sub(Box::new(lit_int(10)), Box::new(lit_int(4)));
        let expr = ShaclExpression::Div(Box::new(inner_sub), Box::new(lit_int(2)));
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/A")).expect("evaluate");
        // Division may return Float or Integer depending on implementation
        let val = match result {
            ShaclValue::Float(f) => f as i64,
            ShaclValue::Integer(n) => n,
            other => panic!("unexpected {other:?}"),
        };
        assert_eq!(val, 3);
    }

    // ---- Chained comparisons -------------------------------------------

    #[test]
    fn test_chained_and_comparisons() {
        // 5 > 1 AND 5 < 10 → true
        let c1 = ShaclExpression::Gt(Box::new(lit_int(5)), Box::new(lit_int(1)));
        let c2 = ShaclExpression::Lt(Box::new(lit_int(5)), Box::new(lit_int(10)));
        let expr = ShaclExpression::And(Box::new(c1), Box::new(c2));
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/A")).expect("evaluate");
        assert_eq!(result, ShaclValue::Boolean(true));
    }

    #[test]
    fn test_or_with_one_false_one_true() {
        let expr = ShaclExpression::Or(Box::new(lit_bool(false)), Box::new(lit_bool(true)));
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/A")).expect("evaluate");
        assert_eq!(result, ShaclValue::Boolean(true));
    }

    #[test]
    fn test_not_true_is_false() {
        let expr = ShaclExpression::Not(Box::new(lit_bool(true)));
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/A")).expect("evaluate");
        assert_eq!(result, ShaclValue::Boolean(false));
    }

    #[test]
    fn test_not_false_is_true() {
        let expr = ShaclExpression::Not(Box::new(lit_bool(false)));
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/A")).expect("evaluate");
        assert_eq!(result, ShaclValue::Boolean(true));
    }

    // ---- Equality ------------------------------------------------------

    #[test]
    fn test_eq_integers_equal() {
        let expr = ShaclExpression::Eq(Box::new(lit_int(42)), Box::new(lit_int(42)));
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/A")).expect("evaluate");
        assert_eq!(result, ShaclValue::Boolean(true));
    }

    #[test]
    fn test_eq_integers_not_equal() {
        let expr = ShaclExpression::Eq(Box::new(lit_int(1)), Box::new(lit_int(2)));
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/A")).expect("evaluate");
        assert_eq!(result, ShaclValue::Boolean(false));
    }

    #[test]
    fn test_ne_integers_different() {
        let expr = ShaclExpression::Ne(Box::new(lit_int(3)), Box::new(lit_int(7)));
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/A")).expect("evaluate");
        assert_eq!(result, ShaclValue::Boolean(true));
    }

    // ---- String operations (UpperCase / LowerCase) ---------------------

    #[test]
    fn test_uppercase_converts_lowercase() {
        let expr = ShaclExpression::UpperCase(Box::new(lit_str("hello world")));
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/A")).expect("evaluate");
        match result {
            ShaclValue::Literal { value, .. } => assert_eq!(value, "HELLO WORLD"),
            other => panic!("expected Literal, got {other:?}"),
        }
    }

    #[test]
    fn test_lowercase_converts_uppercase() {
        let expr = ShaclExpression::LowerCase(Box::new(lit_str("HELLO")));
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/A")).expect("evaluate");
        match result {
            ShaclValue::Literal { value, .. } => assert_eq!(value, "hello"),
            other => panic!("expected Literal, got {other:?}"),
        }
    }

    #[test]
    fn test_strlen_nonempty_string() {
        let expr = ShaclExpression::StrLen(Box::new(lit_str("hello")));
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/A")).expect("evaluate");
        assert_eq!(result, ShaclValue::Integer(5));
    }

    #[test]
    fn test_strlen_empty_string() {
        let expr = ShaclExpression::StrLen(Box::new(lit_str("")));
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/A")).expect("evaluate");
        assert_eq!(result, ShaclValue::Integer(0));
    }

    #[test]
    fn test_concat_two_strings() {
        let expr = ShaclExpression::Concat(vec![lit_str("foo"), lit_str("bar")]);
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/A")).expect("evaluate");
        match result {
            ShaclValue::Literal { value, .. } => assert_eq!(value, "foobar"),
            ShaclValue::Integer(n) => panic!("unexpected integer {n}"),
            other => panic!("expected Literal, got {other:?}"),
        }
    }

    #[test]
    fn test_concat_empty_list() {
        let expr = ShaclExpression::Concat(vec![]);
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/A")).expect("evaluate");
        match result {
            ShaclValue::Literal { value, .. } => assert!(value.is_empty()),
            ShaclValue::Integer(0) => {} // acceptable: some impls return 0 for empty concat
            other => panic!("unexpected: {other:?}"),
        }
    }

    // ---- ExpressionConstraintComponent builder -------------------------

    #[test]
    fn test_builder_with_message_stored() {
        let c = ExpressionConstraintComponent::new(lit_bool(true)).with_message("custom message");
        assert_eq!(c.message.as_deref(), Some("custom message"));
    }

    #[test]
    fn test_builder_deactivated_flag() {
        let c = ExpressionConstraintComponent::new(lit_bool(false)).deactivated();
        let result = c.evaluate(&ctx("http://ex/Alice")).expect("evaluate");
        assert!(
            result.is_valid,
            "deactivated constraint must always be valid"
        );
    }

    #[test]
    fn test_builder_with_message_returned_on_violation() {
        let c =
            ExpressionConstraintComponent::new(lit_bool(false)).with_message("violation occurred");
        let result = c.evaluate(&ctx("http://ex/Alice")).expect("evaluate");
        assert!(!result.is_valid);
        assert_eq!(result.message.as_deref(), Some("violation occurred"));
    }

    // ---- ShaclValue helpers --------------------------------------------

    #[test]
    fn test_shacl_value_boolean_true_eq() {
        assert_eq!(ShaclValue::Boolean(true), ShaclValue::Boolean(true));
    }

    #[test]
    fn test_shacl_value_boolean_false_ne_true() {
        assert_ne!(ShaclValue::Boolean(false), ShaclValue::Boolean(true));
    }

    #[test]
    fn test_shacl_value_null_eq_null() {
        assert_eq!(ShaclValue::Null, ShaclValue::Null);
    }

    #[test]
    fn test_shacl_value_null_is_not_truthy() {
        assert!(!ShaclValue::Null.is_truthy());
    }

    #[test]
    fn test_shacl_value_integer_zero_is_not_truthy() {
        assert!(!ShaclValue::Integer(0).is_truthy());
    }

    #[test]
    fn test_shacl_value_integer_nonzero_is_truthy() {
        assert!(ShaclValue::Integer(42).is_truthy());
    }

    #[test]
    fn test_shacl_value_iri_is_truthy() {
        assert!(ShaclValue::Iri("http://ex/a".to_string()).is_truthy());
    }

    #[test]
    fn test_shacl_value_as_string_integer() {
        assert_eq!(ShaclValue::Integer(99).as_string(), "99");
    }

    #[test]
    fn test_shacl_value_as_string_boolean_true() {
        assert_eq!(ShaclValue::Boolean(true).as_string(), "true");
    }

    #[test]
    fn test_shacl_value_as_string_null_is_empty() {
        assert_eq!(ShaclValue::Null.as_string(), "");
    }

    // ---- Lte / Gte comparisons -----------------------------------------

    #[test]
    fn test_lte_equal_values() {
        let expr = ShaclExpression::Lte(Box::new(lit_int(5)), Box::new(lit_int(5)));
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/A")).expect("evaluate");
        assert_eq!(result, ShaclValue::Boolean(true));
    }

    #[test]
    fn test_gte_equal_values() {
        let expr = ShaclExpression::Gte(Box::new(lit_int(5)), Box::new(lit_int(5)));
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/A")).expect("evaluate");
        assert_eq!(result, ShaclValue::Boolean(true));
    }

    #[test]
    fn test_lte_less_than() {
        let expr = ShaclExpression::Lte(Box::new(lit_int(3)), Box::new(lit_int(7)));
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/A")).expect("evaluate");
        assert_eq!(result, ShaclValue::Boolean(true));
    }

    #[test]
    fn test_gte_greater_than() {
        let expr = ShaclExpression::Gte(Box::new(lit_int(7)), Box::new(lit_int(3)));
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/A")).expect("evaluate");
        assert_eq!(result, ShaclValue::Boolean(true));
    }

    // ---- IsIri / IsLiteral / Datatype ----------------------------------

    #[test]
    fn test_is_iri_true_for_iri() {
        let expr = ShaclExpression::IsIri(Box::new(ShaclExpression::Literal(ShaclValue::Iri(
            "http://example.org/b".to_string(),
        ))));
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/A")).expect("evaluate");
        assert_eq!(result, ShaclValue::Boolean(true));
    }

    #[test]
    fn test_is_iri_false_for_integer() {
        let expr = ShaclExpression::IsIri(Box::new(lit_int(42)));
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/A")).expect("evaluate");
        assert_eq!(result, ShaclValue::Boolean(false));
    }

    #[test]
    fn test_is_literal_true_for_literal_value() {
        let literal_expr = ShaclExpression::Literal(ShaclValue::Literal {
            value: "test".to_string(),
            datatype: None,
            lang: None,
        });
        let expr = ShaclExpression::IsLiteral(Box::new(literal_expr));
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/A")).expect("evaluate");
        assert_eq!(result, ShaclValue::Boolean(true));
    }

    #[test]
    fn test_datatype_integer() {
        let expr = ShaclExpression::Datatype(Box::new(lit_int(42)));
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/A")).expect("evaluate");
        assert_eq!(
            result,
            ShaclValue::Iri("http://www.w3.org/2001/XMLSchema#integer".to_string())
        );
    }

    #[test]
    fn test_datatype_boolean() {
        let expr = ShaclExpression::Datatype(Box::new(lit_bool(true)));
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/A")).expect("evaluate");
        assert_eq!(
            result,
            ShaclValue::Iri("http://www.w3.org/2001/XMLSchema#boolean".to_string())
        );
    }

    #[test]
    fn test_datatype_iri_returns_its_own_type() {
        let expr = ShaclExpression::Datatype(Box::new(ShaclExpression::Literal(ShaclValue::Iri(
            "http://ex/r".to_string(),
        ))));
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/A")).expect("evaluate");
        // IRI has no datatype — implementation specific but should not panic
        assert!(matches!(result, ShaclValue::Iri(_) | ShaclValue::Null));
    }

    // ---- Abs / Floor / Ceil / Round ------------------------------------

    #[test]
    fn test_abs_positive_unchanged() {
        let expr = ShaclExpression::Abs(Box::new(lit_int(7)));
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/A")).expect("evaluate");
        assert_eq!(result, ShaclValue::Integer(7));
    }

    #[test]
    fn test_floor_float() {
        let expr =
            ShaclExpression::Floor(Box::new(ShaclExpression::Literal(ShaclValue::Float(3.9))));
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/A")).expect("evaluate");
        // Accept Float(3.0) or Integer(3)
        let n = match result {
            ShaclValue::Float(f) => f as i64,
            ShaclValue::Integer(n) => n,
            other => panic!("unexpected {other:?}"),
        };
        assert_eq!(n, 3);
    }

    #[test]
    fn test_ceil_float() {
        let expr =
            ShaclExpression::Ceil(Box::new(ShaclExpression::Literal(ShaclValue::Float(3.1))));
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/A")).expect("evaluate");
        let n = match result {
            ShaclValue::Float(f) => f as i64,
            ShaclValue::Integer(n) => n,
            other => panic!("unexpected {other:?}"),
        };
        assert_eq!(n, 4);
    }

    #[test]
    fn test_round_float() {
        let expr =
            ShaclExpression::Round(Box::new(ShaclExpression::Literal(ShaclValue::Float(3.5))));
        let result = ExpressionEvaluator::evaluate(&expr, &ctx("http://ex/A")).expect("evaluate");
        let n = match result {
            ShaclValue::Float(f) => f as i64,
            ShaclValue::Integer(n) => n,
            other => panic!("unexpected {other:?}"),
        };
        // 3.5 rounds to 4 (round half up) or 4 (round half to even)
        assert!(n == 4 || n == 3, "round(3.5) should be 3 or 4, got {n}");
    }

    // ---- ExpressionConstraintResult ------------------------------------

    #[test]
    fn test_result_is_valid_true_for_truthy_expression() {
        let c = ExpressionConstraintComponent::new(lit_int(1));
        let result = c.evaluate(&ctx("http://ex/Alice")).expect("evaluate");
        assert!(result.is_valid);
    }

    #[test]
    fn test_result_is_valid_false_for_falsy_expression() {
        let c = ExpressionConstraintComponent::new(lit_int(0));
        let result = c.evaluate(&ctx("http://ex/Alice")).expect("evaluate");
        assert!(!result.is_valid);
    }

    #[test]
    fn test_result_focus_node_matches_context() {
        let c = ExpressionConstraintComponent::new(lit_bool(true));
        let result = c.evaluate(&ctx("http://ex/MyNode")).expect("evaluate");
        assert_eq!(result.focus_node, "http://ex/MyNode");
    }

    #[test]
    fn test_result_message_none_when_valid() {
        let c = ExpressionConstraintComponent::new(lit_bool(true));
        let result = c.evaluate(&ctx("http://ex/Alice")).expect("evaluate");
        assert!(result.message.is_none());
    }

    #[test]
    fn test_result_default_message_when_no_template() {
        let c = ExpressionConstraintComponent::new(lit_bool(false));
        let result = c.evaluate(&ctx("http://ex/Alice")).expect("evaluate");
        assert!(result.message.is_some());
        assert!(result
            .message
            .as_deref()
            .expect("should succeed")
            .contains("Expression constraint failed"));
    }
}
