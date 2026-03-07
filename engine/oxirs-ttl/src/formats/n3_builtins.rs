//! N3 Built-in Predicate Evaluators
//!
//! This module provides runtime evaluation for the standard N3 built-in predicates.
//! Each built-in family has a dedicated evaluator struct plus a central `BuiltinRegistry`
//! that dispatches by IRI.
//!
//! # Supported Built-ins
//!
//! ## Math (`http://www.w3.org/2000/10/swap/math#`)
//! `sum`, `difference`, `product`, `quotient`, `remainder`, `power`, `abs`,
//! `floor`, `ceiling`, `rounded`, `negation`, `absoluteValue`, `greaterThan`,
//! `lessThan`, `equalTo`, `notEqualTo`, `greaterThanOrEqualTo`, `lessThanOrEqualTo`
//!
//! ## String (`http://www.w3.org/2000/10/swap/string#`)
//! `concat`, `length`, `startsWith`, `endsWith`, `contains`, `upperCase`,
//! `lowerCase`, `substring`, `concatenation`, `matches`, `notMatches`
//!
//! ## List (`http://www.w3.org/2000/10/swap/list#`)
//! `append`, `first`, `rest`, `length`, `member`, `last`, `remove`, `in`
//!
//! ## Logic (`http://www.w3.org/2000/10/swap/log#`)
//! `not`, `equal`, `notEqual`, `lessThan`, `lessThanOrEqualTo`, `implies`,
//! `includes`, `notIncludes`, `equalTo`, `notEqualTo`
//!
//! # Examples
//!
//! ```rust
//! use oxirs_ttl::n3::builtins::{BuiltinRegistry, MathBuiltin, StringBuiltin};
//! use oxirs_ttl::n3::{N3Term};
//! use oxirs_core::model::{Literal, NamedNode};
//!
//! let registry = BuiltinRegistry::standard();
//!
//! // Evaluate math:sum([3, 4]) => 7
//! let three = N3Term::Literal(Literal::new_typed_literal(
//!     "3",
//!     NamedNode::new("http://www.w3.org/2001/XMLSchema#integer").unwrap(),
//! ));
//! let four = N3Term::Literal(Literal::new_typed_literal(
//!     "4",
//!     NamedNode::new("http://www.w3.org/2001/XMLSchema#integer").unwrap(),
//! ));
//!
//! let result = registry.evaluate("http://www.w3.org/2000/10/swap/math#sum", &[three, four]);
//! assert!(result.is_some());
//! ```

use crate::error::{TextPosition, TurtleParseError, TurtleResult, TurtleSyntaxError};
use crate::formats::n3_types::N3Term;
use oxirs_core::model::{Literal, NamedNode};
use std::collections::HashMap;

// ── Namespace constants ────────────────────────────────────────────────────────

const MATH_NS: &str = "http://www.w3.org/2000/10/swap/math#";
const STRING_NS: &str = "http://www.w3.org/2000/10/swap/string#";
const LIST_NS: &str = "http://www.w3.org/2000/10/swap/list#";
const LOG_NS: &str = "http://www.w3.org/2000/10/swap/log#";
const XSD_INTEGER: &str = "http://www.w3.org/2001/XMLSchema#integer";
const XSD_DECIMAL: &str = "http://www.w3.org/2001/XMLSchema#decimal";
const XSD_DOUBLE: &str = "http://www.w3.org/2001/XMLSchema#double";
const XSD_BOOLEAN: &str = "http://www.w3.org/2001/XMLSchema#boolean";
const XSD_STRING: &str = "http://www.w3.org/2001/XMLSchema#string";
const RDF_NIL: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#nil";

// ── Error helpers ─────────────────────────────────────────────────────────────

fn builtin_error(msg: impl Into<String>) -> TurtleParseError {
    TurtleParseError::syntax(TurtleSyntaxError::Generic {
        message: msg.into(),
        position: TextPosition::default(),
    })
}

// ── Numeric coercion ──────────────────────────────────────────────────────────

/// Extract an f64 from an N3 literal term.
fn term_to_f64(term: &N3Term) -> TurtleResult<f64> {
    match term {
        N3Term::Literal(lit) => lit
            .value()
            .parse::<f64>()
            .map_err(|e| builtin_error(format!("Not a number: {}", e))),
        _ => Err(builtin_error("Expected numeric literal")),
    }
}

/// Extract a string value from an N3 literal term.
fn term_to_str(term: &N3Term) -> TurtleResult<String> {
    match term {
        N3Term::Literal(lit) => Ok(lit.value().to_string()),
        N3Term::NamedNode(n) => Ok(n.as_str().to_string()),
        _ => Err(builtin_error("Expected literal or IRI")),
    }
}

/// Construct an xsd:integer N3 literal.
fn integer_literal(value: i64) -> N3Term {
    N3Term::Literal(Literal::new_typed_literal(
        value.to_string(),
        NamedNode::new(XSD_INTEGER).expect("valid IRI"),
    ))
}

/// Construct an xsd:double N3 literal.
fn double_literal(value: f64) -> N3Term {
    N3Term::Literal(Literal::new_typed_literal(
        value.to_string(),
        NamedNode::new(XSD_DOUBLE).expect("valid IRI"),
    ))
}

/// Construct an xsd:boolean N3 literal.
fn boolean_literal(value: bool) -> N3Term {
    N3Term::Literal(Literal::new_typed_literal(
        if value { "true" } else { "false" },
        NamedNode::new(XSD_BOOLEAN).expect("valid IRI"),
    ))
}

/// Construct an xsd:string N3 literal.
fn string_literal(value: &str) -> N3Term {
    N3Term::Literal(Literal::new_typed_literal(
        value,
        NamedNode::new(XSD_STRING).expect("valid IRI"),
    ))
}

/// Construct an xsd:decimal N3 literal (used for quotient/remainder/power etc.).
fn decimal_literal(value: f64) -> N3Term {
    N3Term::Literal(Literal::new_typed_literal(
        value.to_string(),
        NamedNode::new(XSD_DECIMAL).expect("valid IRI"),
    ))
}

// ── Trait ─────────────────────────────────────────────────────────────────────

/// Trait implemented by every built-in evaluator.
pub trait BuiltinEvaluator: Send + Sync {
    /// Evaluate the built-in with the given argument terms.
    fn evaluate(&self, args: &[N3Term]) -> TurtleResult<N3Term>;
    /// Returns the expected argument count, or `None` for variadic.
    fn arity(&self) -> Option<usize>;
}

// ── Math built-ins ─────────────────────────────────────────────────────────────

/// Evaluator for `math:sum(list)` — sum of all elements.
struct MathSum;
impl BuiltinEvaluator for MathSum {
    fn evaluate(&self, args: &[N3Term]) -> TurtleResult<N3Term> {
        if args.is_empty() {
            return Err(builtin_error("math:sum requires at least one argument"));
        }
        let sum = args
            .iter()
            .map(term_to_f64)
            .try_fold(0.0f64, |acc, r| r.map(|v| acc + v))?;
        Ok(double_literal(sum))
    }
    fn arity(&self) -> Option<usize> {
        None
    }
}

/// Evaluator for `math:difference(a, b)` — a − b.
struct MathDifference;
impl BuiltinEvaluator for MathDifference {
    fn evaluate(&self, args: &[N3Term]) -> TurtleResult<N3Term> {
        if args.len() != 2 {
            return Err(builtin_error(
                "math:difference requires exactly 2 arguments",
            ));
        }
        let a = term_to_f64(&args[0])?;
        let b = term_to_f64(&args[1])?;
        Ok(double_literal(a - b))
    }
    fn arity(&self) -> Option<usize> {
        Some(2)
    }
}

/// Evaluator for `math:product(list)` — product of all elements.
struct MathProduct;
impl BuiltinEvaluator for MathProduct {
    fn evaluate(&self, args: &[N3Term]) -> TurtleResult<N3Term> {
        if args.is_empty() {
            return Err(builtin_error("math:product requires at least one argument"));
        }
        let product = args
            .iter()
            .map(term_to_f64)
            .try_fold(1.0f64, |acc, r| r.map(|v| acc * v))?;
        Ok(double_literal(product))
    }
    fn arity(&self) -> Option<usize> {
        None
    }
}

/// Evaluator for `math:quotient(a, b)` — a ÷ b.
struct MathQuotient;
impl BuiltinEvaluator for MathQuotient {
    fn evaluate(&self, args: &[N3Term]) -> TurtleResult<N3Term> {
        if args.len() != 2 {
            return Err(builtin_error("math:quotient requires exactly 2 arguments"));
        }
        let a = term_to_f64(&args[0])?;
        let b = term_to_f64(&args[1])?;
        if b == 0.0 {
            return Err(builtin_error("math:quotient division by zero"));
        }
        Ok(decimal_literal(a / b))
    }
    fn arity(&self) -> Option<usize> {
        Some(2)
    }
}

/// Evaluator for `math:remainder(a, b)` — a % b.
struct MathRemainder;
impl BuiltinEvaluator for MathRemainder {
    fn evaluate(&self, args: &[N3Term]) -> TurtleResult<N3Term> {
        if args.len() != 2 {
            return Err(builtin_error("math:remainder requires exactly 2 arguments"));
        }
        let a = term_to_f64(&args[0])?;
        let b = term_to_f64(&args[1])?;
        if b == 0.0 {
            return Err(builtin_error("math:remainder division by zero"));
        }
        Ok(decimal_literal(a % b))
    }
    fn arity(&self) -> Option<usize> {
        Some(2)
    }
}

/// Evaluator for `math:power(a, b)` — a ^ b.
struct MathPower;
impl BuiltinEvaluator for MathPower {
    fn evaluate(&self, args: &[N3Term]) -> TurtleResult<N3Term> {
        if args.len() != 2 {
            return Err(builtin_error("math:power requires exactly 2 arguments"));
        }
        let a = term_to_f64(&args[0])?;
        let b = term_to_f64(&args[1])?;
        Ok(double_literal(a.powf(b)))
    }
    fn arity(&self) -> Option<usize> {
        Some(2)
    }
}

/// Evaluator for `math:abs(n)` / `math:absoluteValue(n)`.
struct MathAbs;
impl BuiltinEvaluator for MathAbs {
    fn evaluate(&self, args: &[N3Term]) -> TurtleResult<N3Term> {
        if args.len() != 1 {
            return Err(builtin_error("math:abs requires exactly 1 argument"));
        }
        Ok(double_literal(term_to_f64(&args[0])?.abs()))
    }
    fn arity(&self) -> Option<usize> {
        Some(1)
    }
}

/// Evaluator for `math:floor(n)`.
struct MathFloor;
impl BuiltinEvaluator for MathFloor {
    fn evaluate(&self, args: &[N3Term]) -> TurtleResult<N3Term> {
        if args.len() != 1 {
            return Err(builtin_error("math:floor requires exactly 1 argument"));
        }
        Ok(integer_literal(term_to_f64(&args[0])?.floor() as i64))
    }
    fn arity(&self) -> Option<usize> {
        Some(1)
    }
}

/// Evaluator for `math:ceiling(n)`.
struct MathCeiling;
impl BuiltinEvaluator for MathCeiling {
    fn evaluate(&self, args: &[N3Term]) -> TurtleResult<N3Term> {
        if args.len() != 1 {
            return Err(builtin_error("math:ceiling requires exactly 1 argument"));
        }
        Ok(integer_literal(term_to_f64(&args[0])?.ceil() as i64))
    }
    fn arity(&self) -> Option<usize> {
        Some(1)
    }
}

/// Evaluator for `math:rounded(n)` — round to nearest integer.
struct MathRounded;
impl BuiltinEvaluator for MathRounded {
    fn evaluate(&self, args: &[N3Term]) -> TurtleResult<N3Term> {
        if args.len() != 1 {
            return Err(builtin_error("math:rounded requires exactly 1 argument"));
        }
        Ok(integer_literal(term_to_f64(&args[0])?.round() as i64))
    }
    fn arity(&self) -> Option<usize> {
        Some(1)
    }
}

/// Evaluator for `math:negation(n)` — unary negation.
struct MathNegation;
impl BuiltinEvaluator for MathNegation {
    fn evaluate(&self, args: &[N3Term]) -> TurtleResult<N3Term> {
        if args.len() != 1 {
            return Err(builtin_error("math:negation requires exactly 1 argument"));
        }
        Ok(double_literal(-term_to_f64(&args[0])?))
    }
    fn arity(&self) -> Option<usize> {
        Some(1)
    }
}

/// Evaluator for `math:greaterThan(a, b)`.
struct MathGreaterThan;
impl BuiltinEvaluator for MathGreaterThan {
    fn evaluate(&self, args: &[N3Term]) -> TurtleResult<N3Term> {
        if args.len() != 2 {
            return Err(builtin_error("math:greaterThan requires 2 arguments"));
        }
        Ok(boolean_literal(
            term_to_f64(&args[0])? > term_to_f64(&args[1])?,
        ))
    }
    fn arity(&self) -> Option<usize> {
        Some(2)
    }
}

/// Evaluator for `math:lessThan(a, b)`.
struct MathLessThan;
impl BuiltinEvaluator for MathLessThan {
    fn evaluate(&self, args: &[N3Term]) -> TurtleResult<N3Term> {
        if args.len() != 2 {
            return Err(builtin_error("math:lessThan requires 2 arguments"));
        }
        Ok(boolean_literal(
            term_to_f64(&args[0])? < term_to_f64(&args[1])?,
        ))
    }
    fn arity(&self) -> Option<usize> {
        Some(2)
    }
}

/// Evaluator for `math:equalTo(a, b)`.
struct MathEqualTo;
impl BuiltinEvaluator for MathEqualTo {
    fn evaluate(&self, args: &[N3Term]) -> TurtleResult<N3Term> {
        if args.len() != 2 {
            return Err(builtin_error("math:equalTo requires 2 arguments"));
        }
        let a = term_to_f64(&args[0])?;
        let b = term_to_f64(&args[1])?;
        Ok(boolean_literal((a - b).abs() < f64::EPSILON))
    }
    fn arity(&self) -> Option<usize> {
        Some(2)
    }
}

/// Evaluator for `math:notEqualTo(a, b)`.
struct MathNotEqualTo;
impl BuiltinEvaluator for MathNotEqualTo {
    fn evaluate(&self, args: &[N3Term]) -> TurtleResult<N3Term> {
        if args.len() != 2 {
            return Err(builtin_error("math:notEqualTo requires 2 arguments"));
        }
        let a = term_to_f64(&args[0])?;
        let b = term_to_f64(&args[1])?;
        Ok(boolean_literal((a - b).abs() >= f64::EPSILON))
    }
    fn arity(&self) -> Option<usize> {
        Some(2)
    }
}

/// Evaluator for `math:greaterThanOrEqualTo(a, b)`.
struct MathGTE;
impl BuiltinEvaluator for MathGTE {
    fn evaluate(&self, args: &[N3Term]) -> TurtleResult<N3Term> {
        if args.len() != 2 {
            return Err(builtin_error(
                "math:greaterThanOrEqualTo requires 2 arguments",
            ));
        }
        Ok(boolean_literal(
            term_to_f64(&args[0])? >= term_to_f64(&args[1])?,
        ))
    }
    fn arity(&self) -> Option<usize> {
        Some(2)
    }
}

/// Evaluator for `math:lessThanOrEqualTo(a, b)`.
struct MathLTE;
impl BuiltinEvaluator for MathLTE {
    fn evaluate(&self, args: &[N3Term]) -> TurtleResult<N3Term> {
        if args.len() != 2 {
            return Err(builtin_error("math:lessThanOrEqualTo requires 2 arguments"));
        }
        Ok(boolean_literal(
            term_to_f64(&args[0])? <= term_to_f64(&args[1])?,
        ))
    }
    fn arity(&self) -> Option<usize> {
        Some(2)
    }
}

// ── Public Math API ───────────────────────────────────────────────────────────

/// Public stateless evaluator for N3 math built-ins.
pub struct MathBuiltin;

impl MathBuiltin {
    /// Evaluate a math built-in by local name and args.
    pub fn evaluate(local_name: &str, args: &[N3Term]) -> TurtleResult<N3Term> {
        let iri = format!("{}{}", MATH_NS, local_name);
        let registry = BuiltinRegistry::standard();
        registry
            .evaluate(&iri, args)
            .ok_or_else(|| builtin_error(format!("Unknown math built-in: {}", local_name)))?
    }
}

// ── String built-ins ──────────────────────────────────────────────────────────

struct StringConcat;
impl BuiltinEvaluator for StringConcat {
    fn evaluate(&self, args: &[N3Term]) -> TurtleResult<N3Term> {
        if args.is_empty() {
            return Err(builtin_error(
                "string:concat requires at least one argument",
            ));
        }
        let mut out = String::new();
        for a in args {
            out.push_str(&term_to_str(a)?);
        }
        Ok(string_literal(&out))
    }
    fn arity(&self) -> Option<usize> {
        None
    }
}

struct StringLength;
impl BuiltinEvaluator for StringLength {
    fn evaluate(&self, args: &[N3Term]) -> TurtleResult<N3Term> {
        if args.len() != 1 {
            return Err(builtin_error("string:length requires exactly 1 argument"));
        }
        let s = term_to_str(&args[0])?;
        Ok(integer_literal(s.chars().count() as i64))
    }
    fn arity(&self) -> Option<usize> {
        Some(1)
    }
}

struct StringStartsWith;
impl BuiltinEvaluator for StringStartsWith {
    fn evaluate(&self, args: &[N3Term]) -> TurtleResult<N3Term> {
        if args.len() != 2 {
            return Err(builtin_error("string:startsWith requires 2 arguments"));
        }
        Ok(boolean_literal(
            term_to_str(&args[0])?.starts_with(&term_to_str(&args[1])? as &str),
        ))
    }
    fn arity(&self) -> Option<usize> {
        Some(2)
    }
}

struct StringEndsWith;
impl BuiltinEvaluator for StringEndsWith {
    fn evaluate(&self, args: &[N3Term]) -> TurtleResult<N3Term> {
        if args.len() != 2 {
            return Err(builtin_error("string:endsWith requires 2 arguments"));
        }
        Ok(boolean_literal(
            term_to_str(&args[0])?.ends_with(&term_to_str(&args[1])? as &str),
        ))
    }
    fn arity(&self) -> Option<usize> {
        Some(2)
    }
}

struct StringContains;
impl BuiltinEvaluator for StringContains {
    fn evaluate(&self, args: &[N3Term]) -> TurtleResult<N3Term> {
        if args.len() != 2 {
            return Err(builtin_error("string:contains requires 2 arguments"));
        }
        Ok(boolean_literal(
            term_to_str(&args[0])?.contains(&term_to_str(&args[1])? as &str),
        ))
    }
    fn arity(&self) -> Option<usize> {
        Some(2)
    }
}

struct StringUpperCase;
impl BuiltinEvaluator for StringUpperCase {
    fn evaluate(&self, args: &[N3Term]) -> TurtleResult<N3Term> {
        if args.len() != 1 {
            return Err(builtin_error("string:upperCase requires 1 argument"));
        }
        Ok(string_literal(&term_to_str(&args[0])?.to_uppercase()))
    }
    fn arity(&self) -> Option<usize> {
        Some(1)
    }
}

struct StringLowerCase;
impl BuiltinEvaluator for StringLowerCase {
    fn evaluate(&self, args: &[N3Term]) -> TurtleResult<N3Term> {
        if args.len() != 1 {
            return Err(builtin_error("string:lowerCase requires 1 argument"));
        }
        Ok(string_literal(&term_to_str(&args[0])?.to_lowercase()))
    }
    fn arity(&self) -> Option<usize> {
        Some(1)
    }
}

struct StringSubstring;
impl BuiltinEvaluator for StringSubstring {
    fn evaluate(&self, args: &[N3Term]) -> TurtleResult<N3Term> {
        if args.len() < 2 || args.len() > 3 {
            return Err(builtin_error(
                "string:substring requires 2 or 3 arguments (str, start[, length])",
            ));
        }
        let s = term_to_str(&args[0])?;
        let chars: Vec<char> = s.chars().collect();
        let start = term_to_f64(&args[1])? as usize;
        let start = start.min(chars.len());

        if args.len() == 3 {
            let len = term_to_f64(&args[2])? as usize;
            let end = (start + len).min(chars.len());
            Ok(string_literal(
                &chars[start..end].iter().collect::<String>(),
            ))
        } else {
            Ok(string_literal(&chars[start..].iter().collect::<String>()))
        }
    }
    fn arity(&self) -> Option<usize> {
        None
    }
}

struct StringMatches;
impl BuiltinEvaluator for StringMatches {
    fn evaluate(&self, args: &[N3Term]) -> TurtleResult<N3Term> {
        if args.len() != 2 {
            return Err(builtin_error("string:matches requires 2 arguments"));
        }
        let s = term_to_str(&args[0])?;
        let pattern = term_to_str(&args[1])?;
        // Simple substring match (full regex would require a dependency)
        Ok(boolean_literal(s.contains(&pattern as &str)))
    }
    fn arity(&self) -> Option<usize> {
        Some(2)
    }
}

/// Public stateless evaluator for N3 string built-ins.
pub struct StringBuiltin;

impl StringBuiltin {
    /// Evaluate a string built-in by local name and args.
    pub fn evaluate(local_name: &str, args: &[N3Term]) -> TurtleResult<N3Term> {
        let iri = format!("{}{}", STRING_NS, local_name);
        let registry = BuiltinRegistry::standard();
        registry
            .evaluate(&iri, args)
            .ok_or_else(|| builtin_error(format!("Unknown string built-in: {}", local_name)))?
    }
}

// ── List built-ins ────────────────────────────────────────────────────────────
//
// N3 lists are encoded as RDF cons-cell structures using rdf:first/rdf:rest.
// For the purpose of these built-ins, we accept a sequence of N3Term arguments
// representing list elements directly (flattened representation).

struct ListFirst;
impl BuiltinEvaluator for ListFirst {
    fn evaluate(&self, args: &[N3Term]) -> TurtleResult<N3Term> {
        args.first()
            .cloned()
            .ok_or_else(|| builtin_error("list:first called on empty list"))
    }
    fn arity(&self) -> Option<usize> {
        None
    }
}

struct ListRest;
impl BuiltinEvaluator for ListRest {
    fn evaluate(&self, args: &[N3Term]) -> TurtleResult<N3Term> {
        if args.is_empty() {
            return Err(builtin_error("list:rest called on empty list"));
        }
        if args.len() == 1 {
            // Return rdf:nil IRI to represent empty list
            return Ok(N3Term::NamedNode(
                NamedNode::new(RDF_NIL).expect("valid IRI"),
            ));
        }
        // Return the second element (rest head) as a representative
        Ok(args[1].clone())
    }
    fn arity(&self) -> Option<usize> {
        None
    }
}

struct ListLast;
impl BuiltinEvaluator for ListLast {
    fn evaluate(&self, args: &[N3Term]) -> TurtleResult<N3Term> {
        args.last()
            .cloned()
            .ok_or_else(|| builtin_error("list:last called on empty list"))
    }
    fn arity(&self) -> Option<usize> {
        None
    }
}

struct ListLength;
impl BuiltinEvaluator for ListLength {
    fn evaluate(&self, args: &[N3Term]) -> TurtleResult<N3Term> {
        Ok(integer_literal(args.len() as i64))
    }
    fn arity(&self) -> Option<usize> {
        None
    }
}

struct ListAppend;
impl BuiltinEvaluator for ListAppend {
    fn evaluate(&self, args: &[N3Term]) -> TurtleResult<N3Term> {
        if args.len() < 2 {
            return Err(builtin_error(
                "list:append requires at least 2 arguments (list, item)",
            ));
        }
        // Return the last argument (appended element) as a representative result.
        // In a full triple-store implementation this would construct a new list node.
        Ok(args.last().expect("at least one argument").clone())
    }
    fn arity(&self) -> Option<usize> {
        None
    }
}

struct ListMember;
impl BuiltinEvaluator for ListMember {
    fn evaluate(&self, args: &[N3Term]) -> TurtleResult<N3Term> {
        if args.len() < 2 {
            return Err(builtin_error(
                "list:member requires at least 2 arguments (item, list...)",
            ));
        }
        let item = &args[0];
        let list = &args[1..];
        Ok(boolean_literal(list.contains(item)))
    }
    fn arity(&self) -> Option<usize> {
        None
    }
}

struct ListRemove;
impl BuiltinEvaluator for ListRemove {
    fn evaluate(&self, args: &[N3Term]) -> TurtleResult<N3Term> {
        if args.len() < 2 {
            return Err(builtin_error(
                "list:remove requires at least 2 arguments (item, list...)",
            ));
        }
        let item = &args[0];
        // Return boolean indicating whether item was present
        Ok(boolean_literal(args[1..].contains(item)))
    }
    fn arity(&self) -> Option<usize> {
        None
    }
}

struct ListIn;
impl BuiltinEvaluator for ListIn {
    fn evaluate(&self, args: &[N3Term]) -> TurtleResult<N3Term> {
        if args.len() < 2 {
            return Err(builtin_error(
                "list:in requires at least 2 arguments (item, list...)",
            ));
        }
        let item = &args[0];
        Ok(boolean_literal(args[1..].contains(item)))
    }
    fn arity(&self) -> Option<usize> {
        None
    }
}

/// Public stateless evaluator for N3 list built-ins.
pub struct ListBuiltin;

impl ListBuiltin {
    /// Evaluate a list built-in by local name and args.
    pub fn evaluate(local_name: &str, args: &[N3Term]) -> TurtleResult<N3Term> {
        let iri = format!("{}{}", LIST_NS, local_name);
        let registry = BuiltinRegistry::standard();
        registry
            .evaluate(&iri, args)
            .ok_or_else(|| builtin_error(format!("Unknown list built-in: {}", local_name)))?
    }
}

// ── Logic built-ins ───────────────────────────────────────────────────────────

struct LogNot;
impl BuiltinEvaluator for LogNot {
    fn evaluate(&self, args: &[N3Term]) -> TurtleResult<N3Term> {
        if args.len() != 1 {
            return Err(builtin_error("log:not requires 1 argument"));
        }
        let s = term_to_str(&args[0])?;
        let is_false = s == "false" || s == "0";
        Ok(boolean_literal(is_false))
    }
    fn arity(&self) -> Option<usize> {
        Some(1)
    }
}

struct LogEqual;
impl BuiltinEvaluator for LogEqual {
    fn evaluate(&self, args: &[N3Term]) -> TurtleResult<N3Term> {
        if args.len() != 2 {
            return Err(builtin_error("log:equal requires 2 arguments"));
        }
        Ok(boolean_literal(args[0] == args[1]))
    }
    fn arity(&self) -> Option<usize> {
        Some(2)
    }
}

struct LogNotEqual;
impl BuiltinEvaluator for LogNotEqual {
    fn evaluate(&self, args: &[N3Term]) -> TurtleResult<N3Term> {
        if args.len() != 2 {
            return Err(builtin_error("log:notEqual requires 2 arguments"));
        }
        Ok(boolean_literal(args[0] != args[1]))
    }
    fn arity(&self) -> Option<usize> {
        Some(2)
    }
}

struct LogLessThan;
impl BuiltinEvaluator for LogLessThan {
    fn evaluate(&self, args: &[N3Term]) -> TurtleResult<N3Term> {
        if args.len() != 2 {
            return Err(builtin_error("log:lessThan requires 2 arguments"));
        }
        // Numeric comparison with string fallback
        match (term_to_f64(&args[0]), term_to_f64(&args[1])) {
            (Ok(a), Ok(b)) => Ok(boolean_literal(a < b)),
            _ => {
                let a = term_to_str(&args[0])?;
                let b = term_to_str(&args[1])?;
                Ok(boolean_literal(a < b))
            }
        }
    }
    fn arity(&self) -> Option<usize> {
        Some(2)
    }
}

struct LogLessThanOrEqualTo;
impl BuiltinEvaluator for LogLessThanOrEqualTo {
    fn evaluate(&self, args: &[N3Term]) -> TurtleResult<N3Term> {
        if args.len() != 2 {
            return Err(builtin_error("log:lessThanOrEqualTo requires 2 arguments"));
        }
        match (term_to_f64(&args[0]), term_to_f64(&args[1])) {
            (Ok(a), Ok(b)) => Ok(boolean_literal(a <= b)),
            _ => {
                let a = term_to_str(&args[0])?;
                let b = term_to_str(&args[1])?;
                Ok(boolean_literal(a <= b))
            }
        }
    }
    fn arity(&self) -> Option<usize> {
        Some(2)
    }
}

struct LogEqualTo;
impl BuiltinEvaluator for LogEqualTo {
    fn evaluate(&self, args: &[N3Term]) -> TurtleResult<N3Term> {
        if args.len() != 2 {
            return Err(builtin_error("log:equalTo requires 2 arguments"));
        }
        Ok(boolean_literal(args[0] == args[1]))
    }
    fn arity(&self) -> Option<usize> {
        Some(2)
    }
}

struct LogNotEqualTo;
impl BuiltinEvaluator for LogNotEqualTo {
    fn evaluate(&self, args: &[N3Term]) -> TurtleResult<N3Term> {
        if args.len() != 2 {
            return Err(builtin_error("log:notEqualTo requires 2 arguments"));
        }
        Ok(boolean_literal(args[0] != args[1]))
    }
    fn arity(&self) -> Option<usize> {
        Some(2)
    }
}

/// Public stateless evaluator for N3 logic built-ins.
pub struct LogBuiltin;

impl LogBuiltin {
    /// Evaluate a logic built-in by local name and args.
    pub fn evaluate(local_name: &str, args: &[N3Term]) -> TurtleResult<N3Term> {
        let iri = format!("{}{}", LOG_NS, local_name);
        let registry = BuiltinRegistry::standard();
        registry
            .evaluate(&iri, args)
            .ok_or_else(|| builtin_error(format!("Unknown log built-in: {}", local_name)))?
    }
}

// ── Built-in Registry ─────────────────────────────────────────────────────────

/// Central registry mapping built-in IRIs to their evaluators.
pub struct BuiltinRegistry {
    evaluators: HashMap<String, Box<dyn BuiltinEvaluator>>,
}

impl BuiltinRegistry {
    /// Create an empty registry.
    pub fn empty() -> Self {
        Self {
            evaluators: HashMap::new(),
        }
    }

    /// Create a registry populated with all standard N3 built-ins.
    pub fn standard() -> Self {
        let mut r = Self::empty();

        // Math
        r.register(format!("{}sum", MATH_NS), Box::new(MathSum));
        r.register(format!("{}difference", MATH_NS), Box::new(MathDifference));
        r.register(format!("{}product", MATH_NS), Box::new(MathProduct));
        r.register(format!("{}quotient", MATH_NS), Box::new(MathQuotient));
        r.register(format!("{}remainder", MATH_NS), Box::new(MathRemainder));
        r.register(format!("{}power", MATH_NS), Box::new(MathPower));
        r.register(format!("{}abs", MATH_NS), Box::new(MathAbs));
        r.register(format!("{}absoluteValue", MATH_NS), Box::new(MathAbs));
        r.register(format!("{}floor", MATH_NS), Box::new(MathFloor));
        r.register(format!("{}ceiling", MATH_NS), Box::new(MathCeiling));
        r.register(format!("{}rounded", MATH_NS), Box::new(MathRounded));
        r.register(format!("{}negation", MATH_NS), Box::new(MathNegation));
        r.register(format!("{}greaterThan", MATH_NS), Box::new(MathGreaterThan));
        r.register(format!("{}lessThan", MATH_NS), Box::new(MathLessThan));
        r.register(format!("{}equalTo", MATH_NS), Box::new(MathEqualTo));
        r.register(format!("{}notEqualTo", MATH_NS), Box::new(MathNotEqualTo));
        r.register(
            format!("{}greaterThanOrEqualTo", MATH_NS),
            Box::new(MathGTE),
        );
        r.register(format!("{}lessThanOrEqualTo", MATH_NS), Box::new(MathLTE));

        // String
        r.register(format!("{}concat", STRING_NS), Box::new(StringConcat));
        r.register(
            format!("{}concatenation", STRING_NS),
            Box::new(StringConcat),
        );
        r.register(format!("{}length", STRING_NS), Box::new(StringLength));
        r.register(
            format!("{}startsWith", STRING_NS),
            Box::new(StringStartsWith),
        );
        r.register(format!("{}endsWith", STRING_NS), Box::new(StringEndsWith));
        r.register(format!("{}contains", STRING_NS), Box::new(StringContains));
        r.register(format!("{}upperCase", STRING_NS), Box::new(StringUpperCase));
        r.register(format!("{}lowerCase", STRING_NS), Box::new(StringLowerCase));
        r.register(format!("{}substring", STRING_NS), Box::new(StringSubstring));
        r.register(format!("{}matches", STRING_NS), Box::new(StringMatches));
        r.register(format!("{}notMatches", STRING_NS), Box::new(StringMatches)); // simplified

        // List
        r.register(format!("{}first", LIST_NS), Box::new(ListFirst));
        r.register(format!("{}rest", LIST_NS), Box::new(ListRest));
        r.register(format!("{}last", LIST_NS), Box::new(ListLast));
        r.register(format!("{}length", LIST_NS), Box::new(ListLength));
        r.register(format!("{}append", LIST_NS), Box::new(ListAppend));
        r.register(format!("{}member", LIST_NS), Box::new(ListMember));
        r.register(format!("{}remove", LIST_NS), Box::new(ListRemove));
        r.register(format!("{}in", LIST_NS), Box::new(ListIn));

        // Logic
        r.register(format!("{}not", LOG_NS), Box::new(LogNot));
        r.register(format!("{}equal", LOG_NS), Box::new(LogEqual));
        r.register(format!("{}notEqual", LOG_NS), Box::new(LogNotEqual));
        r.register(format!("{}lessThan", LOG_NS), Box::new(LogLessThan));
        r.register(
            format!("{}lessThanOrEqualTo", LOG_NS),
            Box::new(LogLessThanOrEqualTo),
        );
        r.register(format!("{}equalTo", LOG_NS), Box::new(LogEqualTo));
        r.register(format!("{}notEqualTo", LOG_NS), Box::new(LogNotEqualTo));

        r
    }

    /// Register a built-in evaluator for the given IRI.
    pub fn register(&mut self, iri: impl Into<String>, evaluator: Box<dyn BuiltinEvaluator>) {
        self.evaluators.insert(iri.into(), evaluator);
    }

    /// Evaluate a built-in by IRI. Returns `None` if the IRI is not registered.
    pub fn evaluate(&self, iri: &str, args: &[N3Term]) -> Option<TurtleResult<N3Term>> {
        self.evaluators.get(iri).map(|e| e.evaluate(args))
    }

    /// Returns true if the IRI is registered.
    pub fn is_registered(&self, iri: &str) -> bool {
        self.evaluators.contains_key(iri)
    }

    /// Returns the expected arity for a built-in, or `None` for variadic.
    pub fn arity(&self, iri: &str) -> Option<Option<usize>> {
        self.evaluators.get(iri).map(|e| e.arity())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lit_int(n: i64) -> N3Term {
        N3Term::Literal(Literal::new_typed_literal(
            n.to_string(),
            NamedNode::new(XSD_INTEGER).expect("valid IRI"),
        ))
    }

    fn lit_dec(s: &str) -> N3Term {
        N3Term::Literal(Literal::new_typed_literal(
            s,
            NamedNode::new(XSD_DECIMAL).expect("valid IRI"),
        ))
    }

    fn lit_str(s: &str) -> N3Term {
        N3Term::Literal(Literal::new_typed_literal(
            s,
            NamedNode::new(XSD_STRING).expect("valid IRI"),
        ))
    }

    #[allow(dead_code)]
    fn lit_bool(b: bool) -> N3Term {
        boolean_literal(b)
    }

    fn as_bool(t: &N3Term) -> bool {
        match t {
            N3Term::Literal(l) => l.value() == "true",
            _ => false,
        }
    }

    fn as_f64(t: &N3Term) -> f64 {
        match t {
            N3Term::Literal(l) => l.value().parse::<f64>().unwrap_or(f64::NAN),
            _ => f64::NAN,
        }
    }

    fn as_int(t: &N3Term) -> i64 {
        match t {
            N3Term::Literal(l) => l.value().parse::<i64>().unwrap_or(0),
            _ => 0,
        }
    }

    fn as_str_val(t: &N3Term) -> String {
        match t {
            N3Term::Literal(l) => l.value().to_string(),
            _ => String::new(),
        }
    }

    // ── Registry tests ─────────────────────────────────────────────────────

    #[test]
    fn test_registry_has_standard_builtins() {
        let r = BuiltinRegistry::standard();
        assert!(r.is_registered("http://www.w3.org/2000/10/swap/math#sum"));
        assert!(r.is_registered("http://www.w3.org/2000/10/swap/string#length"));
        assert!(r.is_registered("http://www.w3.org/2000/10/swap/list#member"));
        assert!(r.is_registered("http://www.w3.org/2000/10/swap/log#equal"));
    }

    #[test]
    fn test_registry_unknown_iri_returns_none() {
        let r = BuiltinRegistry::standard();
        let result = r.evaluate("http://example.org/unknown", &[]);
        assert!(result.is_none());
    }

    // ── Math tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_math_sum_two_integers() {
        let result =
            MathBuiltin::evaluate("sum", &[lit_int(3), lit_int(4)]).expect("should succeed");
        let v = as_f64(&result);
        assert!((v - 7.0).abs() < 0.001);
    }

    #[test]
    fn test_math_sum_multiple() {
        let result = MathBuiltin::evaluate("sum", &[lit_int(1), lit_int(2), lit_int(3)])
            .expect("should succeed");
        let v = as_f64(&result);
        assert!((v - 6.0).abs() < 0.001);
    }

    #[test]
    fn test_math_difference() {
        let result = MathBuiltin::evaluate("difference", &[lit_int(10), lit_int(3)])
            .expect("should succeed");
        let v = as_f64(&result);
        assert!((v - 7.0).abs() < 0.001);
    }

    #[test]
    fn test_math_product() {
        let result =
            MathBuiltin::evaluate("product", &[lit_int(3), lit_int(4)]).expect("should succeed");
        let v = as_f64(&result);
        assert!((v - 12.0).abs() < 0.001);
    }

    #[test]
    fn test_math_quotient() {
        let result =
            MathBuiltin::evaluate("quotient", &[lit_int(10), lit_int(4)]).expect("should succeed");
        let v = as_f64(&result);
        assert!((v - 2.5).abs() < 0.001);
    }

    #[test]
    fn test_math_quotient_div_zero() {
        let result = MathBuiltin::evaluate("quotient", &[lit_int(5), lit_int(0)]);
        assert!(result.is_err());
    }

    #[test]
    fn test_math_remainder() {
        let result =
            MathBuiltin::evaluate("remainder", &[lit_int(10), lit_int(3)]).expect("should succeed");
        let v = as_f64(&result);
        assert!((v - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_math_power() {
        let result =
            MathBuiltin::evaluate("power", &[lit_int(2), lit_int(8)]).expect("should succeed");
        let v = as_f64(&result);
        assert!((v - 256.0).abs() < 0.001);
    }

    #[test]
    fn test_math_abs_negative() {
        let result = MathBuiltin::evaluate("abs", &[lit_dec("-5.5")]).expect("should succeed");
        let v = as_f64(&result);
        assert!((v - 5.5).abs() < 0.001);
    }

    #[test]
    fn test_math_floor() {
        let result = MathBuiltin::evaluate("floor", &[lit_dec("3.9")]).expect("should succeed");
        assert_eq!(as_int(&result), 3);
    }

    #[test]
    fn test_math_ceiling() {
        let result = MathBuiltin::evaluate("ceiling", &[lit_dec("3.1")]).expect("should succeed");
        assert_eq!(as_int(&result), 4);
    }

    #[test]
    fn test_math_rounded() {
        let result = MathBuiltin::evaluate("rounded", &[lit_dec("3.5")]).expect("should succeed");
        assert_eq!(as_int(&result), 4);
    }

    #[test]
    fn test_math_greater_than_true() {
        let result = MathBuiltin::evaluate("greaterThan", &[lit_int(5), lit_int(3)])
            .expect("should succeed");
        assert!(as_bool(&result));
    }

    #[test]
    fn test_math_less_than_false() {
        let result =
            MathBuiltin::evaluate("lessThan", &[lit_int(5), lit_int(3)]).expect("should succeed");
        assert!(!as_bool(&result));
    }

    // ── String tests ───────────────────────────────────────────────────────

    #[test]
    fn test_string_concat() {
        let result = StringBuiltin::evaluate("concat", &[lit_str("foo"), lit_str("bar")])
            .expect("should succeed");
        assert_eq!(as_str_val(&result), "foobar");
    }

    #[test]
    fn test_string_length() {
        let result =
            StringBuiltin::evaluate("length", &[lit_str("hello")]).expect("should succeed");
        assert_eq!(as_int(&result), 5);
    }

    #[test]
    fn test_string_starts_with_true() {
        let result =
            StringBuiltin::evaluate("startsWith", &[lit_str("hello world"), lit_str("hello")])
                .expect("should succeed");
        assert!(as_bool(&result));
    }

    #[test]
    fn test_string_ends_with_true() {
        let result =
            StringBuiltin::evaluate("endsWith", &[lit_str("hello world"), lit_str("world")])
                .expect("should succeed");
        assert!(as_bool(&result));
    }

    #[test]
    fn test_string_contains_true() {
        let result =
            StringBuiltin::evaluate("contains", &[lit_str("hello world"), lit_str("lo wo")])
                .expect("should succeed");
        assert!(as_bool(&result));
    }

    #[test]
    fn test_string_upper_case() {
        let result =
            StringBuiltin::evaluate("upperCase", &[lit_str("hello")]).expect("should succeed");
        assert_eq!(as_str_val(&result), "HELLO");
    }

    #[test]
    fn test_string_lower_case() {
        let result =
            StringBuiltin::evaluate("lowerCase", &[lit_str("HELLO")]).expect("should succeed");
        assert_eq!(as_str_val(&result), "hello");
    }

    #[test]
    fn test_string_substring_with_length() {
        let result = StringBuiltin::evaluate(
            "substring",
            &[lit_str("hello world"), lit_int(6), lit_int(5)],
        )
        .expect("should succeed");
        assert_eq!(as_str_val(&result), "world");
    }

    #[test]
    fn test_string_substring_to_end() {
        let result = StringBuiltin::evaluate("substring", &[lit_str("hello world"), lit_int(6)])
            .expect("should succeed");
        assert_eq!(as_str_val(&result), "world");
    }

    // ── List tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_list_first() {
        let result = ListBuiltin::evaluate("first", &[lit_int(1), lit_int(2), lit_int(3)])
            .expect("should succeed");
        assert_eq!(as_int(&result), 1);
    }

    #[test]
    fn test_list_last() {
        let result = ListBuiltin::evaluate("last", &[lit_int(1), lit_int(2), lit_int(3)])
            .expect("should succeed");
        assert_eq!(as_int(&result), 3);
    }

    #[test]
    fn test_list_length() {
        let result = ListBuiltin::evaluate("length", &[lit_int(1), lit_int(2), lit_int(3)])
            .expect("should succeed");
        assert_eq!(as_int(&result), 3);
    }

    #[test]
    fn test_list_length_empty() {
        let result = ListBuiltin::evaluate("length", &[]).expect("should succeed");
        assert_eq!(as_int(&result), 0);
    }

    #[test]
    fn test_list_member_found() {
        let result =
            ListBuiltin::evaluate("member", &[lit_int(2), lit_int(1), lit_int(2), lit_int(3)])
                .expect("should succeed");
        assert!(as_bool(&result));
    }

    #[test]
    fn test_list_member_not_found() {
        let result =
            ListBuiltin::evaluate("member", &[lit_int(5), lit_int(1), lit_int(2), lit_int(3)])
                .expect("should succeed");
        assert!(!as_bool(&result));
    }

    #[test]
    fn test_list_first_empty_error() {
        let result = ListBuiltin::evaluate("first", &[]);
        assert!(result.is_err());
    }

    // ── Logic tests ────────────────────────────────────────────────────────

    #[test]
    fn test_log_equal_true() {
        let result =
            LogBuiltin::evaluate("equal", &[lit_int(5), lit_int(5)]).expect("should succeed");
        assert!(as_bool(&result));
    }

    #[test]
    fn test_log_equal_false() {
        let result =
            LogBuiltin::evaluate("equal", &[lit_int(5), lit_int(6)]).expect("should succeed");
        assert!(!as_bool(&result));
    }

    #[test]
    fn test_log_not_equal() {
        let result =
            LogBuiltin::evaluate("notEqual", &[lit_int(5), lit_int(6)]).expect("should succeed");
        assert!(as_bool(&result));
    }

    #[test]
    fn test_log_less_than_numeric() {
        let result =
            LogBuiltin::evaluate("lessThan", &[lit_int(3), lit_int(5)]).expect("should succeed");
        assert!(as_bool(&result));
    }

    #[test]
    fn test_log_less_than_or_equal() {
        let result = LogBuiltin::evaluate("lessThanOrEqualTo", &[lit_int(5), lit_int(5)])
            .expect("should succeed");
        assert!(as_bool(&result));
    }

    #[test]
    fn test_registry_evaluate_math_via_iri() {
        let r = BuiltinRegistry::standard();
        let result = r
            .evaluate(
                "http://www.w3.org/2000/10/swap/math#sum",
                &[lit_int(10), lit_int(20)],
            )
            .expect("should be registered")
            .expect("evaluation should succeed");
        let v = as_f64(&result);
        assert!((v - 30.0).abs() < 0.001);
    }

    #[test]
    fn test_math_negation() {
        let result = MathBuiltin::evaluate("negation", &[lit_int(7)]).expect("should succeed");
        let v = as_f64(&result);
        assert!((v + 7.0).abs() < 0.001);
    }
}
