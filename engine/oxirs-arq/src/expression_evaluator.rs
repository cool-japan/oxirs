/// SPARQL 1.1 expression/filter evaluator.
///
/// Evaluates SPARQL filter expressions over a binding environment, supporting
/// arithmetic, comparison, logical, type-checking, and string operations.
use std::cmp::Ordering;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Value type
// ---------------------------------------------------------------------------

/// An RDF value that can appear in a SPARQL binding or expression result.
#[derive(Debug, Clone, PartialEq)]
pub enum RdfValue {
    /// An IRI resource.
    Iri(String),
    /// A typed or plain literal.
    Literal { value: String, datatype: String },
    /// A blank node.
    BlankNode(String),
    /// A boolean effective value.
    Boolean(bool),
    /// An integer numeric value.
    Integer(i64),
    /// A double/float numeric value.
    Double(f64),
    /// A plain string (shorthand for xsd:string literal).
    String(String),
}

impl RdfValue {
    /// Produce a human-readable string representation.
    pub fn display_string(&self) -> std::string::String {
        match self {
            RdfValue::Iri(s) => s.clone(),
            RdfValue::Literal { value, .. } => value.clone(),
            RdfValue::BlankNode(id) => format!("_:{id}"),
            RdfValue::Boolean(b) => b.to_string(),
            RdfValue::Integer(i) => i.to_string(),
            RdfValue::Double(d) => d.to_string(),
            RdfValue::String(s) => s.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Expression AST
// ---------------------------------------------------------------------------

/// A SPARQL 1.1 filter/project expression.
#[derive(Debug, Clone)]
pub enum Expr {
    /// Constant value.
    Const(RdfValue),
    /// Variable reference.
    Var(String),
    // Arithmetic
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
    // Comparison
    Eq(Box<Expr>, Box<Expr>),
    Ne(Box<Expr>, Box<Expr>),
    Lt(Box<Expr>, Box<Expr>),
    Le(Box<Expr>, Box<Expr>),
    Gt(Box<Expr>, Box<Expr>),
    Ge(Box<Expr>, Box<Expr>),
    // Logical
    And(Box<Expr>, Box<Expr>),
    Or(Box<Expr>, Box<Expr>),
    Not(Box<Expr>),
    // Type tests
    IsIri(Box<Expr>),
    IsLiteral(Box<Expr>),
    IsBlank(Box<Expr>),
    /// True when the variable is bound in the current solution.
    Bound(String),
    // Accessors
    Str(Box<Expr>),
    Lang(Box<Expr>),
    Datatype(Box<Expr>),
    // String functions
    Concat(Vec<Expr>),
    StrLen(Box<Expr>),
    Regex(Box<Expr>, Box<Expr>),
    // Control
    If(Box<Expr>, Box<Expr>, Box<Expr>),
}

// ---------------------------------------------------------------------------
// Evaluation error
// ---------------------------------------------------------------------------

/// Errors that can occur during expression evaluation.
#[derive(Debug, PartialEq)]
pub enum EvalError {
    /// A variable referenced in the expression is not bound.
    UnboundVariable(std::string::String),
    /// A value has an unexpected type for the operation.
    TypeError(std::string::String),
    /// Division by zero attempted.
    DivisionByZero,
}

impl std::fmt::Display for EvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EvalError::UnboundVariable(v) => write!(f, "Unbound variable: {v}"),
            EvalError::TypeError(msg) => write!(f, "Type error: {msg}"),
            EvalError::DivisionByZero => write!(f, "Division by zero"),
        }
    }
}

impl std::error::Error for EvalError {}

// ---------------------------------------------------------------------------
// Evaluator
// ---------------------------------------------------------------------------

/// Stateless evaluator for SPARQL 1.1 filter expressions.
pub struct ExprEvaluator;

impl ExprEvaluator {
    /// Evaluate `expr` in the context of the given variable `bindings`.
    ///
    /// Returns the resulting [`RdfValue`] or an [`EvalError`].
    pub fn eval(
        expr: &Expr,
        bindings: &HashMap<std::string::String, RdfValue>,
    ) -> Result<RdfValue, EvalError> {
        match expr {
            Expr::Const(v) => Ok(v.clone()),

            Expr::Var(name) => bindings
                .get(name)
                .cloned()
                .ok_or_else(|| EvalError::UnboundVariable(name.clone())),

            // ---- Arithmetic ----
            Expr::Add(a, b) => {
                let lhs = Self::eval(a, bindings)?;
                let rhs = Self::eval(b, bindings)?;
                Self::arith_op(&lhs, &rhs, |a, b| a + b, |a, b| a + b)
            }
            Expr::Sub(a, b) => {
                let lhs = Self::eval(a, bindings)?;
                let rhs = Self::eval(b, bindings)?;
                Self::arith_op(&lhs, &rhs, |a, b| a - b, |a, b| a - b)
            }
            Expr::Mul(a, b) => {
                let lhs = Self::eval(a, bindings)?;
                let rhs = Self::eval(b, bindings)?;
                Self::arith_op(&lhs, &rhs, |a, b| a * b, |a, b| a * b)
            }
            Expr::Div(a, b) => {
                let lhs = Self::eval(a, bindings)?;
                let rhs = Self::eval(b, bindings)?;
                match (&lhs, &rhs) {
                    (RdfValue::Integer(_), RdfValue::Integer(0)) => Err(EvalError::DivisionByZero),
                    (RdfValue::Double(_), RdfValue::Double(d)) if *d == 0.0 => {
                        Err(EvalError::DivisionByZero)
                    }
                    _ => Self::arith_op(&lhs, &rhs, |a, b| a / b, |a, b| a / b),
                }
            }

            // ---- Comparison ----
            Expr::Eq(a, b) => {
                let lhs = Self::eval(a, bindings)?;
                let rhs = Self::eval(b, bindings)?;
                Ok(RdfValue::Boolean(lhs == rhs))
            }
            Expr::Ne(a, b) => {
                let lhs = Self::eval(a, bindings)?;
                let rhs = Self::eval(b, bindings)?;
                Ok(RdfValue::Boolean(lhs != rhs))
            }
            Expr::Lt(a, b) => {
                let lhs = Self::eval(a, bindings)?;
                let rhs = Self::eval(b, bindings)?;
                Self::cmp_op(&lhs, &rhs, Ordering::Less)
            }
            Expr::Le(a, b) => {
                let lhs = Self::eval(a, bindings)?;
                let rhs = Self::eval(b, bindings)?;
                match Self::compare(&lhs, &rhs) {
                    Some(Ordering::Less) | Some(Ordering::Equal) => Ok(RdfValue::Boolean(true)),
                    Some(_) => Ok(RdfValue::Boolean(false)),
                    None => Err(EvalError::TypeError(format!(
                        "Cannot compare {lhs:?} <= {rhs:?}"
                    ))),
                }
            }
            Expr::Gt(a, b) => {
                let lhs = Self::eval(a, bindings)?;
                let rhs = Self::eval(b, bindings)?;
                Self::cmp_op(&lhs, &rhs, Ordering::Greater)
            }
            Expr::Ge(a, b) => {
                let lhs = Self::eval(a, bindings)?;
                let rhs = Self::eval(b, bindings)?;
                match Self::compare(&lhs, &rhs) {
                    Some(Ordering::Greater) | Some(Ordering::Equal) => Ok(RdfValue::Boolean(true)),
                    Some(_) => Ok(RdfValue::Boolean(false)),
                    None => Err(EvalError::TypeError(format!(
                        "Cannot compare {lhs:?} >= {rhs:?}"
                    ))),
                }
            }

            // ---- Logical ----
            Expr::And(a, b) => {
                let lhs = Self::eval(a, bindings)?;
                let rhs = Self::eval(b, bindings)?;
                Ok(RdfValue::Boolean(
                    Self::is_truthy(&lhs) && Self::is_truthy(&rhs),
                ))
            }
            Expr::Or(a, b) => {
                let lhs = Self::eval(a, bindings)?;
                let rhs = Self::eval(b, bindings)?;
                Ok(RdfValue::Boolean(
                    Self::is_truthy(&lhs) || Self::is_truthy(&rhs),
                ))
            }
            Expr::Not(inner) => {
                let val = Self::eval(inner, bindings)?;
                Ok(RdfValue::Boolean(!Self::is_truthy(&val)))
            }

            // ---- Type tests ----
            Expr::IsIri(inner) => {
                let val = Self::eval(inner, bindings)?;
                Ok(RdfValue::Boolean(matches!(val, RdfValue::Iri(_))))
            }
            Expr::IsLiteral(inner) => {
                let val = Self::eval(inner, bindings)?;
                Ok(RdfValue::Boolean(matches!(
                    val,
                    RdfValue::Literal { .. }
                        | RdfValue::Integer(_)
                        | RdfValue::Double(_)
                        | RdfValue::Boolean(_)
                        | RdfValue::String(_)
                )))
            }
            Expr::IsBlank(inner) => {
                let val = Self::eval(inner, bindings)?;
                Ok(RdfValue::Boolean(matches!(val, RdfValue::BlankNode(_))))
            }
            Expr::Bound(varname) => Ok(RdfValue::Boolean(bindings.contains_key(varname))),

            // ---- Accessors ----
            Expr::Str(inner) => {
                let val = Self::eval(inner, bindings)?;
                Ok(RdfValue::String(val.display_string()))
            }
            Expr::Lang(inner) => {
                let val = Self::eval(inner, bindings)?;
                match &val {
                    RdfValue::Literal { datatype, .. } if datatype.starts_with("@lang:") => Ok(
                        RdfValue::String(datatype.trim_start_matches("@lang:").to_string()),
                    ),
                    _ => Ok(RdfValue::String(std::string::String::new())),
                }
            }
            Expr::Datatype(inner) => {
                let val = Self::eval(inner, bindings)?;
                match &val {
                    RdfValue::Literal { datatype, .. } => Ok(RdfValue::Iri(datatype.clone())),
                    RdfValue::Integer(_) => Ok(RdfValue::Iri(
                        "http://www.w3.org/2001/XMLSchema#integer".to_string(),
                    )),
                    RdfValue::Double(_) => Ok(RdfValue::Iri(
                        "http://www.w3.org/2001/XMLSchema#double".to_string(),
                    )),
                    RdfValue::Boolean(_) => Ok(RdfValue::Iri(
                        "http://www.w3.org/2001/XMLSchema#boolean".to_string(),
                    )),
                    RdfValue::String(_) => Ok(RdfValue::Iri(
                        "http://www.w3.org/2001/XMLSchema#string".to_string(),
                    )),
                    _ => Err(EvalError::TypeError(format!(
                        "datatype() applied to non-literal: {val:?}"
                    ))),
                }
            }

            // ---- String functions ----
            Expr::Concat(parts) => {
                let mut result = std::string::String::new();
                for part in parts {
                    let v = Self::eval(part, bindings)?;
                    result.push_str(&v.display_string());
                }
                Ok(RdfValue::String(result))
            }
            Expr::StrLen(inner) => {
                let val = Self::eval(inner, bindings)?;
                let s = val.display_string();
                Ok(RdfValue::Integer(s.chars().count() as i64))
            }
            Expr::Regex(text_expr, pattern_expr) => {
                let text_val = Self::eval(text_expr, bindings)?;
                let pattern_val = Self::eval(pattern_expr, bindings)?;
                let text = text_val.display_string();
                let pattern = pattern_val.display_string();
                // Simple substring / prefix / exact match — no full regex engine dep
                let matched = text.contains(pattern.as_str());
                Ok(RdfValue::Boolean(matched))
            }

            // ---- Control ----
            Expr::If(cond, then_expr, else_expr) => {
                let cond_val = Self::eval(cond, bindings)?;
                if Self::is_truthy(&cond_val) {
                    Self::eval(then_expr, bindings)
                } else {
                    Self::eval(else_expr, bindings)
                }
            }
        }
    }

    /// Return the effective boolean value of a SPARQL term.
    pub fn is_truthy(val: &RdfValue) -> bool {
        match val {
            RdfValue::Boolean(b) => *b,
            RdfValue::Integer(i) => *i != 0,
            RdfValue::Double(d) => *d != 0.0 && !d.is_nan(),
            RdfValue::String(s) => !s.is_empty(),
            RdfValue::Literal { value, .. } => !value.is_empty(),
            RdfValue::Iri(_) => true,
            RdfValue::BlankNode(_) => true,
        }
    }

    /// Coerce an `RdfValue` to `f64` for numeric operations.
    pub fn coerce_numeric(val: &RdfValue) -> Option<f64> {
        match val {
            RdfValue::Integer(i) => Some(*i as f64),
            RdfValue::Double(d) => Some(*d),
            RdfValue::Literal { value, datatype } => {
                if datatype.contains("integer")
                    || datatype.contains("int")
                    || datatype.contains("decimal")
                    || datatype.contains("float")
                    || datatype.contains("double")
                {
                    value.trim().parse::<f64>().ok()
                } else {
                    None
                }
            }
            RdfValue::String(s) => s.trim().parse::<f64>().ok(),
            _ => None,
        }
    }

    /// Compare two `RdfValue`s.  Returns `None` when they cannot be compared.
    pub fn compare(a: &RdfValue, b: &RdfValue) -> Option<Ordering> {
        match (a, b) {
            (RdfValue::Integer(x), RdfValue::Integer(y)) => Some(x.cmp(y)),
            (RdfValue::Double(x), RdfValue::Double(y)) => x.partial_cmp(y),
            (RdfValue::Integer(x), RdfValue::Double(y)) => (*x as f64).partial_cmp(y),
            (RdfValue::Double(x), RdfValue::Integer(y)) => x.partial_cmp(&(*y as f64)),
            (RdfValue::String(x), RdfValue::String(y)) => Some(x.cmp(y)),
            (RdfValue::Iri(x), RdfValue::Iri(y)) => Some(x.cmp(y)),
            (RdfValue::Boolean(x), RdfValue::Boolean(y)) => Some(x.cmp(y)),
            (RdfValue::Literal { value: v1, .. }, RdfValue::Literal { value: v2, .. }) => {
                // Try numeric first, fall back to lexicographic
                match (Self::coerce_numeric(a), Self::coerce_numeric(b)) {
                    (Some(n1), Some(n2)) => n1.partial_cmp(&n2),
                    _ => Some(v1.cmp(v2)),
                }
            }
            _ => {
                // Cross-type: try numeric coercion
                match (Self::coerce_numeric(a), Self::coerce_numeric(b)) {
                    (Some(n1), Some(n2)) => n1.partial_cmp(&n2),
                    _ => None,
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Perform an arithmetic operation, promoting types as needed.
    fn arith_op(
        lhs: &RdfValue,
        rhs: &RdfValue,
        int_op: impl Fn(i64, i64) -> i64,
        dbl_op: impl Fn(f64, f64) -> f64,
    ) -> Result<RdfValue, EvalError> {
        match (lhs, rhs) {
            (RdfValue::Integer(a), RdfValue::Integer(b)) => Ok(RdfValue::Integer(int_op(*a, *b))),
            (RdfValue::Double(a), RdfValue::Double(b)) => Ok(RdfValue::Double(dbl_op(*a, *b))),
            (RdfValue::Integer(a), RdfValue::Double(b)) => {
                Ok(RdfValue::Double(dbl_op(*a as f64, *b)))
            }
            (RdfValue::Double(a), RdfValue::Integer(b)) => {
                Ok(RdfValue::Double(dbl_op(*a, *b as f64)))
            }
            _ => match (Self::coerce_numeric(lhs), Self::coerce_numeric(rhs)) {
                (Some(a), Some(b)) => Ok(RdfValue::Double(dbl_op(a, b))),
                _ => Err(EvalError::TypeError(format!(
                    "Arithmetic not applicable to {lhs:?} and {rhs:?}"
                ))),
            },
        }
    }

    /// Perform a strict-ordering comparison (exactly `expected`).
    fn cmp_op(lhs: &RdfValue, rhs: &RdfValue, expected: Ordering) -> Result<RdfValue, EvalError> {
        match Self::compare(lhs, rhs) {
            Some(ord) => Ok(RdfValue::Boolean(ord == expected)),
            None => Err(EvalError::TypeError(format!(
                "Cannot compare {lhs:?} and {rhs:?}"
            ))),
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn bindings(pairs: &[(&str, RdfValue)]) -> HashMap<std::string::String, RdfValue> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect()
    }

    // --- Const / Var ---
    #[test]
    fn test_const_integer() {
        let e = Expr::Const(RdfValue::Integer(42));
        let result = ExprEvaluator::eval(&e, &HashMap::new()).unwrap();
        assert_eq!(result, RdfValue::Integer(42));
    }

    #[test]
    fn test_const_double() {
        let e = Expr::Const(RdfValue::Double(2.71));
        let result = ExprEvaluator::eval(&e, &HashMap::new()).unwrap();
        assert!(matches!(result, RdfValue::Double(d) if (d - 2.71).abs() < 1e-9));
    }

    #[test]
    fn test_var_bound() {
        let b = bindings(&[("x", RdfValue::Integer(5))]);
        let e = Expr::Var("x".to_string());
        assert_eq!(ExprEvaluator::eval(&e, &b).unwrap(), RdfValue::Integer(5));
    }

    #[test]
    fn test_var_unbound() {
        let e = Expr::Var("missing".to_string());
        assert!(matches!(
            ExprEvaluator::eval(&e, &HashMap::new()).unwrap_err(),
            EvalError::UnboundVariable(_)
        ));
    }

    // --- Arithmetic ---
    #[test]
    fn test_add_integers() {
        let e = Expr::Add(
            Box::new(Expr::Const(RdfValue::Integer(3))),
            Box::new(Expr::Const(RdfValue::Integer(4))),
        );
        assert_eq!(
            ExprEvaluator::eval(&e, &HashMap::new()).unwrap(),
            RdfValue::Integer(7)
        );
    }

    #[test]
    fn test_add_doubles() {
        let e = Expr::Add(
            Box::new(Expr::Const(RdfValue::Double(1.5))),
            Box::new(Expr::Const(RdfValue::Double(2.5))),
        );
        let result = ExprEvaluator::eval(&e, &HashMap::new()).unwrap();
        assert!(matches!(result, RdfValue::Double(d) if (d - 4.0).abs() < 1e-9));
    }

    #[test]
    fn test_add_int_double_promotion() {
        let e = Expr::Add(
            Box::new(Expr::Const(RdfValue::Integer(2))),
            Box::new(Expr::Const(RdfValue::Double(0.5))),
        );
        let result = ExprEvaluator::eval(&e, &HashMap::new()).unwrap();
        assert!(matches!(result, RdfValue::Double(d) if (d - 2.5).abs() < 1e-9));
    }

    #[test]
    fn test_sub_integers() {
        let e = Expr::Sub(
            Box::new(Expr::Const(RdfValue::Integer(10))),
            Box::new(Expr::Const(RdfValue::Integer(3))),
        );
        assert_eq!(
            ExprEvaluator::eval(&e, &HashMap::new()).unwrap(),
            RdfValue::Integer(7)
        );
    }

    #[test]
    fn test_mul_integers() {
        let e = Expr::Mul(
            Box::new(Expr::Const(RdfValue::Integer(6))),
            Box::new(Expr::Const(RdfValue::Integer(7))),
        );
        assert_eq!(
            ExprEvaluator::eval(&e, &HashMap::new()).unwrap(),
            RdfValue::Integer(42)
        );
    }

    #[test]
    fn test_div_integers() {
        let e = Expr::Div(
            Box::new(Expr::Const(RdfValue::Integer(20))),
            Box::new(Expr::Const(RdfValue::Integer(4))),
        );
        assert_eq!(
            ExprEvaluator::eval(&e, &HashMap::new()).unwrap(),
            RdfValue::Integer(5)
        );
    }

    #[test]
    fn test_div_by_zero() {
        let e = Expr::Div(
            Box::new(Expr::Const(RdfValue::Integer(1))),
            Box::new(Expr::Const(RdfValue::Integer(0))),
        );
        assert_eq!(
            ExprEvaluator::eval(&e, &HashMap::new()).unwrap_err(),
            EvalError::DivisionByZero
        );
    }

    #[test]
    fn test_div_double_by_zero() {
        let e = Expr::Div(
            Box::new(Expr::Const(RdfValue::Double(1.0))),
            Box::new(Expr::Const(RdfValue::Double(0.0))),
        );
        assert_eq!(
            ExprEvaluator::eval(&e, &HashMap::new()).unwrap_err(),
            EvalError::DivisionByZero
        );
    }

    // --- Comparison ---
    #[test]
    fn test_eq_integers_true() {
        let e = Expr::Eq(
            Box::new(Expr::Const(RdfValue::Integer(5))),
            Box::new(Expr::Const(RdfValue::Integer(5))),
        );
        assert_eq!(
            ExprEvaluator::eval(&e, &HashMap::new()).unwrap(),
            RdfValue::Boolean(true)
        );
    }

    #[test]
    fn test_eq_integers_false() {
        let e = Expr::Eq(
            Box::new(Expr::Const(RdfValue::Integer(5))),
            Box::new(Expr::Const(RdfValue::Integer(6))),
        );
        assert_eq!(
            ExprEvaluator::eval(&e, &HashMap::new()).unwrap(),
            RdfValue::Boolean(false)
        );
    }

    #[test]
    fn test_ne() {
        let e = Expr::Ne(
            Box::new(Expr::Const(RdfValue::Integer(1))),
            Box::new(Expr::Const(RdfValue::Integer(2))),
        );
        assert_eq!(
            ExprEvaluator::eval(&e, &HashMap::new()).unwrap(),
            RdfValue::Boolean(true)
        );
    }

    #[test]
    fn test_lt_true() {
        let e = Expr::Lt(
            Box::new(Expr::Const(RdfValue::Integer(3))),
            Box::new(Expr::Const(RdfValue::Integer(5))),
        );
        assert_eq!(
            ExprEvaluator::eval(&e, &HashMap::new()).unwrap(),
            RdfValue::Boolean(true)
        );
    }

    #[test]
    fn test_lt_false() {
        let e = Expr::Lt(
            Box::new(Expr::Const(RdfValue::Integer(5))),
            Box::new(Expr::Const(RdfValue::Integer(3))),
        );
        assert_eq!(
            ExprEvaluator::eval(&e, &HashMap::new()).unwrap(),
            RdfValue::Boolean(false)
        );
    }

    #[test]
    fn test_le_equal() {
        let e = Expr::Le(
            Box::new(Expr::Const(RdfValue::Integer(5))),
            Box::new(Expr::Const(RdfValue::Integer(5))),
        );
        assert_eq!(
            ExprEvaluator::eval(&e, &HashMap::new()).unwrap(),
            RdfValue::Boolean(true)
        );
    }

    #[test]
    fn test_gt_true() {
        let e = Expr::Gt(
            Box::new(Expr::Const(RdfValue::Integer(10))),
            Box::new(Expr::Const(RdfValue::Integer(5))),
        );
        assert_eq!(
            ExprEvaluator::eval(&e, &HashMap::new()).unwrap(),
            RdfValue::Boolean(true)
        );
    }

    #[test]
    fn test_ge_equal() {
        let e = Expr::Ge(
            Box::new(Expr::Const(RdfValue::Integer(5))),
            Box::new(Expr::Const(RdfValue::Integer(5))),
        );
        assert_eq!(
            ExprEvaluator::eval(&e, &HashMap::new()).unwrap(),
            RdfValue::Boolean(true)
        );
    }

    // --- Logical ---
    #[test]
    fn test_and_true() {
        let e = Expr::And(
            Box::new(Expr::Const(RdfValue::Boolean(true))),
            Box::new(Expr::Const(RdfValue::Boolean(true))),
        );
        assert_eq!(
            ExprEvaluator::eval(&e, &HashMap::new()).unwrap(),
            RdfValue::Boolean(true)
        );
    }

    #[test]
    fn test_and_false() {
        let e = Expr::And(
            Box::new(Expr::Const(RdfValue::Boolean(true))),
            Box::new(Expr::Const(RdfValue::Boolean(false))),
        );
        assert_eq!(
            ExprEvaluator::eval(&e, &HashMap::new()).unwrap(),
            RdfValue::Boolean(false)
        );
    }

    #[test]
    fn test_or_false_true() {
        let e = Expr::Or(
            Box::new(Expr::Const(RdfValue::Boolean(false))),
            Box::new(Expr::Const(RdfValue::Boolean(true))),
        );
        assert_eq!(
            ExprEvaluator::eval(&e, &HashMap::new()).unwrap(),
            RdfValue::Boolean(true)
        );
    }

    #[test]
    fn test_not_true() {
        let e = Expr::Not(Box::new(Expr::Const(RdfValue::Boolean(true))));
        assert_eq!(
            ExprEvaluator::eval(&e, &HashMap::new()).unwrap(),
            RdfValue::Boolean(false)
        );
    }

    #[test]
    fn test_not_false() {
        let e = Expr::Not(Box::new(Expr::Const(RdfValue::Boolean(false))));
        assert_eq!(
            ExprEvaluator::eval(&e, &HashMap::new()).unwrap(),
            RdfValue::Boolean(true)
        );
    }

    // --- Type tests ---
    #[test]
    fn test_is_iri() {
        let e = Expr::IsIri(Box::new(Expr::Const(RdfValue::Iri(
            "http://example.org/".to_string(),
        ))));
        assert_eq!(
            ExprEvaluator::eval(&e, &HashMap::new()).unwrap(),
            RdfValue::Boolean(true)
        );
    }

    #[test]
    fn test_is_iri_false() {
        let e = Expr::IsIri(Box::new(Expr::Const(RdfValue::Integer(1))));
        assert_eq!(
            ExprEvaluator::eval(&e, &HashMap::new()).unwrap(),
            RdfValue::Boolean(false)
        );
    }

    #[test]
    fn test_is_literal_integer() {
        let e = Expr::IsLiteral(Box::new(Expr::Const(RdfValue::Integer(7))));
        assert_eq!(
            ExprEvaluator::eval(&e, &HashMap::new()).unwrap(),
            RdfValue::Boolean(true)
        );
    }

    #[test]
    fn test_is_literal_false_for_iri() {
        let e = Expr::IsLiteral(Box::new(Expr::Const(RdfValue::Iri(
            "http://x.org/".to_string(),
        ))));
        assert_eq!(
            ExprEvaluator::eval(&e, &HashMap::new()).unwrap(),
            RdfValue::Boolean(false)
        );
    }

    #[test]
    fn test_is_blank() {
        let e = Expr::IsBlank(Box::new(Expr::Const(RdfValue::BlankNode("b1".to_string()))));
        assert_eq!(
            ExprEvaluator::eval(&e, &HashMap::new()).unwrap(),
            RdfValue::Boolean(true)
        );
    }

    #[test]
    fn test_is_blank_false() {
        let e = Expr::IsBlank(Box::new(Expr::Const(RdfValue::Iri(
            "http://x.org/".to_string(),
        ))));
        assert_eq!(
            ExprEvaluator::eval(&e, &HashMap::new()).unwrap(),
            RdfValue::Boolean(false)
        );
    }

    #[test]
    fn test_bound_true() {
        let b = bindings(&[("x", RdfValue::Integer(0))]);
        let e = Expr::Bound("x".to_string());
        assert_eq!(
            ExprEvaluator::eval(&e, &b).unwrap(),
            RdfValue::Boolean(true)
        );
    }

    #[test]
    fn test_bound_false() {
        let e = Expr::Bound("y".to_string());
        assert_eq!(
            ExprEvaluator::eval(&e, &HashMap::new()).unwrap(),
            RdfValue::Boolean(false)
        );
    }

    // --- Accessors ---
    #[test]
    fn test_str_on_iri() {
        let e = Expr::Str(Box::new(Expr::Const(RdfValue::Iri(
            "http://example.org/foo".to_string(),
        ))));
        assert_eq!(
            ExprEvaluator::eval(&e, &HashMap::new()).unwrap(),
            RdfValue::String("http://example.org/foo".to_string())
        );
    }

    #[test]
    fn test_str_on_integer() {
        let e = Expr::Str(Box::new(Expr::Const(RdfValue::Integer(42))));
        assert_eq!(
            ExprEvaluator::eval(&e, &HashMap::new()).unwrap(),
            RdfValue::String("42".to_string())
        );
    }

    #[test]
    fn test_datatype_integer() {
        let e = Expr::Datatype(Box::new(Expr::Const(RdfValue::Integer(1))));
        let result = ExprEvaluator::eval(&e, &HashMap::new()).unwrap();
        assert!(matches!(result, RdfValue::Iri(s) if s.contains("integer")));
    }

    #[test]
    fn test_datatype_double() {
        let e = Expr::Datatype(Box::new(Expr::Const(RdfValue::Double(1.0))));
        let result = ExprEvaluator::eval(&e, &HashMap::new()).unwrap();
        assert!(matches!(result, RdfValue::Iri(s) if s.contains("double")));
    }

    #[test]
    fn test_datatype_boolean() {
        let e = Expr::Datatype(Box::new(Expr::Const(RdfValue::Boolean(true))));
        let result = ExprEvaluator::eval(&e, &HashMap::new()).unwrap();
        assert!(matches!(result, RdfValue::Iri(s) if s.contains("boolean")));
    }

    #[test]
    fn test_lang_no_lang() {
        let e = Expr::Lang(Box::new(Expr::Const(RdfValue::Literal {
            value: "hello".to_string(),
            datatype: "http://www.w3.org/2001/XMLSchema#string".to_string(),
        })));
        assert_eq!(
            ExprEvaluator::eval(&e, &HashMap::new()).unwrap(),
            RdfValue::String(std::string::String::new())
        );
    }

    // --- String functions ---
    #[test]
    fn test_concat_two_strings() {
        let e = Expr::Concat(vec![
            Expr::Const(RdfValue::String("Hello".to_string())),
            Expr::Const(RdfValue::String(", World!".to_string())),
        ]);
        assert_eq!(
            ExprEvaluator::eval(&e, &HashMap::new()).unwrap(),
            RdfValue::String("Hello, World!".to_string())
        );
    }

    #[test]
    fn test_concat_empty() {
        let e = Expr::Concat(vec![]);
        assert_eq!(
            ExprEvaluator::eval(&e, &HashMap::new()).unwrap(),
            RdfValue::String(std::string::String::new())
        );
    }

    #[test]
    fn test_strlen() {
        let e = Expr::StrLen(Box::new(Expr::Const(RdfValue::String("hello".to_string()))));
        assert_eq!(
            ExprEvaluator::eval(&e, &HashMap::new()).unwrap(),
            RdfValue::Integer(5)
        );
    }

    #[test]
    fn test_strlen_empty() {
        let e = Expr::StrLen(Box::new(Expr::Const(RdfValue::String(
            std::string::String::new(),
        ))));
        assert_eq!(
            ExprEvaluator::eval(&e, &HashMap::new()).unwrap(),
            RdfValue::Integer(0)
        );
    }

    #[test]
    fn test_regex_match() {
        let e = Expr::Regex(
            Box::new(Expr::Const(RdfValue::String("foobar".to_string()))),
            Box::new(Expr::Const(RdfValue::String("bar".to_string()))),
        );
        assert_eq!(
            ExprEvaluator::eval(&e, &HashMap::new()).unwrap(),
            RdfValue::Boolean(true)
        );
    }

    #[test]
    fn test_regex_no_match() {
        let e = Expr::Regex(
            Box::new(Expr::Const(RdfValue::String("foobar".to_string()))),
            Box::new(Expr::Const(RdfValue::String("xyz".to_string()))),
        );
        assert_eq!(
            ExprEvaluator::eval(&e, &HashMap::new()).unwrap(),
            RdfValue::Boolean(false)
        );
    }

    // --- Control ---
    #[test]
    fn test_if_true_branch() {
        let e = Expr::If(
            Box::new(Expr::Const(RdfValue::Boolean(true))),
            Box::new(Expr::Const(RdfValue::Integer(1))),
            Box::new(Expr::Const(RdfValue::Integer(2))),
        );
        assert_eq!(
            ExprEvaluator::eval(&e, &HashMap::new()).unwrap(),
            RdfValue::Integer(1)
        );
    }

    #[test]
    fn test_if_false_branch() {
        let e = Expr::If(
            Box::new(Expr::Const(RdfValue::Boolean(false))),
            Box::new(Expr::Const(RdfValue::Integer(1))),
            Box::new(Expr::Const(RdfValue::Integer(2))),
        );
        assert_eq!(
            ExprEvaluator::eval(&e, &HashMap::new()).unwrap(),
            RdfValue::Integer(2)
        );
    }

    // --- is_truthy ---
    #[test]
    fn test_is_truthy_bool_true() {
        assert!(ExprEvaluator::is_truthy(&RdfValue::Boolean(true)));
    }

    #[test]
    fn test_is_truthy_bool_false() {
        assert!(!ExprEvaluator::is_truthy(&RdfValue::Boolean(false)));
    }

    #[test]
    fn test_is_truthy_integer_zero() {
        assert!(!ExprEvaluator::is_truthy(&RdfValue::Integer(0)));
    }

    #[test]
    fn test_is_truthy_integer_nonzero() {
        assert!(ExprEvaluator::is_truthy(&RdfValue::Integer(42)));
    }

    #[test]
    fn test_is_truthy_double_nan() {
        assert!(!ExprEvaluator::is_truthy(&RdfValue::Double(f64::NAN)));
    }

    #[test]
    fn test_is_truthy_empty_string() {
        assert!(!ExprEvaluator::is_truthy(&RdfValue::String(
            std::string::String::new()
        )));
    }

    #[test]
    fn test_is_truthy_nonempty_string() {
        assert!(ExprEvaluator::is_truthy(&RdfValue::String("x".to_string())));
    }

    #[test]
    fn test_is_truthy_iri() {
        assert!(ExprEvaluator::is_truthy(&RdfValue::Iri(
            "http://x/".to_string()
        )));
    }

    // --- coerce_numeric ---
    #[test]
    fn test_coerce_numeric_integer() {
        assert_eq!(
            ExprEvaluator::coerce_numeric(&RdfValue::Integer(7)),
            Some(7.0)
        );
    }

    #[test]
    fn test_coerce_numeric_double() {
        assert_eq!(
            ExprEvaluator::coerce_numeric(&RdfValue::Double(2.5)),
            Some(2.5)
        );
    }

    #[test]
    #[allow(clippy::approx_constant)]
    fn test_coerce_numeric_string_number() {
        assert_eq!(
            ExprEvaluator::coerce_numeric(&RdfValue::String("3.14".to_string())),
            Some(3.14)
        );
    }

    #[test]
    fn test_coerce_numeric_string_non_number() {
        assert_eq!(
            ExprEvaluator::coerce_numeric(&RdfValue::String("abc".to_string())),
            None
        );
    }

    #[test]
    fn test_coerce_numeric_iri() {
        assert_eq!(
            ExprEvaluator::coerce_numeric(&RdfValue::Iri("http://x/".to_string())),
            None
        );
    }

    // --- compare ---
    #[test]
    fn test_compare_integers_less() {
        assert_eq!(
            ExprEvaluator::compare(&RdfValue::Integer(1), &RdfValue::Integer(2)),
            Some(Ordering::Less)
        );
    }

    #[test]
    fn test_compare_integers_equal() {
        assert_eq!(
            ExprEvaluator::compare(&RdfValue::Integer(5), &RdfValue::Integer(5)),
            Some(Ordering::Equal)
        );
    }

    #[test]
    fn test_compare_strings() {
        assert_eq!(
            ExprEvaluator::compare(
                &RdfValue::String("abc".to_string()),
                &RdfValue::String("abd".to_string())
            ),
            Some(Ordering::Less)
        );
    }

    #[test]
    fn test_compare_incompatible_types() {
        assert_eq!(
            ExprEvaluator::compare(
                &RdfValue::Iri("http://x/".to_string()),
                &RdfValue::BlankNode("b1".to_string())
            ),
            None
        );
    }

    // --- Complex nested expressions ---
    #[test]
    fn test_nested_arithmetic() {
        // (2 + 3) * 4 == 20
        let e = Expr::Mul(
            Box::new(Expr::Add(
                Box::new(Expr::Const(RdfValue::Integer(2))),
                Box::new(Expr::Const(RdfValue::Integer(3))),
            )),
            Box::new(Expr::Const(RdfValue::Integer(4))),
        );
        assert_eq!(
            ExprEvaluator::eval(&e, &HashMap::new()).unwrap(),
            RdfValue::Integer(20)
        );
    }

    #[test]
    fn test_filter_with_variables() {
        let b = bindings(&[
            ("age", RdfValue::Integer(25)),
            ("limit", RdfValue::Integer(18)),
        ]);
        // age >= limit
        let e = Expr::Ge(
            Box::new(Expr::Var("age".to_string())),
            Box::new(Expr::Var("limit".to_string())),
        );
        assert_eq!(
            ExprEvaluator::eval(&e, &b).unwrap(),
            RdfValue::Boolean(true)
        );
    }

    #[test]
    fn test_logical_complex() {
        // (1 < 2) AND (3 > 1)
        let e = Expr::And(
            Box::new(Expr::Lt(
                Box::new(Expr::Const(RdfValue::Integer(1))),
                Box::new(Expr::Const(RdfValue::Integer(2))),
            )),
            Box::new(Expr::Gt(
                Box::new(Expr::Const(RdfValue::Integer(3))),
                Box::new(Expr::Const(RdfValue::Integer(1))),
            )),
        );
        assert_eq!(
            ExprEvaluator::eval(&e, &HashMap::new()).unwrap(),
            RdfValue::Boolean(true)
        );
    }

    #[test]
    fn test_type_error_on_arith() {
        let e = Expr::Add(
            Box::new(Expr::Const(RdfValue::Iri("http://x/".to_string()))),
            Box::new(Expr::Const(RdfValue::Integer(1))),
        );
        assert!(matches!(
            ExprEvaluator::eval(&e, &HashMap::new()).unwrap_err(),
            EvalError::TypeError(_)
        ));
    }
}
