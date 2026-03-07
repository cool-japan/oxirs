//! SPARQL Expression Compiler and Evaluator with LRU Caching
//!
//! This module implements a two-phase pipeline:
//!   1. `ExprCompiler::compile` — parses an expression string into `CompiledExpr`
//!   2. `ExprCompiler::evaluate` — evaluates a `CompiledExpr` against variable bindings
//!
//! An `ExprCache` wraps the compiler with an LRU cache (VecDeque + HashMap) to avoid
//! re-parsing frequently used expressions.
//!
//! # Supported syntax
//!
//! - Integer and floating-point literals (`42`, `3.14`)
//! - Double-quoted string literals (`"hello"`)
//! - IRI references (`<http://example.org/>`)
//! - Variable references (`?varName`)
//! - Unary minus (`-expr`) and logical NOT (`!expr`)
//! - Binary arithmetic: `+`, `-`, `*`, `/`
//! - Binary comparison: `=`, `!=`, `<`, `<=`, `>`, `>=`
//! - Binary logical: `&&`, `||`
//! - `IF(cond, then, else)`
//! - SPARQL built-in calls: `BOUND`, `ISIRI`, `ISLITERAL`, `ISBLANK`,
//!   `STR`, `LANG`, `DATATYPE`, `COALESCE`
//! - Arbitrary function calls: `FUNC(arg1, arg2, ...)`

use std::collections::{HashMap, VecDeque};
use std::fmt;

// ─── Value ─────────────────────────────────────────────────────────────────

/// A runtime value produced by evaluating a SPARQL expression.
#[derive(Debug, Clone, PartialEq)]
pub enum ExprValue {
    /// Boolean result (e.g. from comparison or BOUND)
    Bool(bool),
    /// Integer numeric value
    Integer(i64),
    /// Double-precision floating-point value
    Double(f64),
    /// Plain or language-tagged string literal
    Str(String),
    /// IRI value
    Iri(String),
    /// Blank-node identifier
    Blank(String),
    /// Unbound variable (no value in binding map)
    Unbound,
}

impl fmt::Display for ExprValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bool(b) => write!(f, "{b}"),
            Self::Integer(i) => write!(f, "{i}"),
            Self::Double(d) => write!(f, "{d}"),
            Self::Str(s) => write!(f, "\"{s}\""),
            Self::Iri(i) => write!(f, "<{i}>"),
            Self::Blank(b) => write!(f, "_:{b}"),
            Self::Unbound => write!(f, "UNBOUND"),
        }
    }
}

// ─── Binary operators ───────────────────────────────────────────────────────

/// Binary operators supported in compiled expressions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BinOp {
    /// Addition
    Add,
    /// Subtraction
    Sub,
    /// Multiplication
    Mul,
    /// Division
    Div,
    /// Equality
    Eq,
    /// Inequality
    Ne,
    /// Less-than
    Lt,
    /// Less-than-or-equal
    Le,
    /// Greater-than
    Gt,
    /// Greater-than-or-equal
    Ge,
    /// Logical AND
    And,
    /// Logical OR
    Or,
}

// ─── Compiled expression ────────────────────────────────────────────────────

/// A compiled SPARQL expression tree.
#[derive(Debug, Clone)]
pub enum CompiledExpr {
    /// A literal value (string representation stored, parsed at eval time)
    Literal(String),
    /// A variable reference (without the leading `?`)
    Variable(String),
    /// An IRI reference (without angle brackets)
    IriRef(String),
    /// Unary arithmetic negation
    Neg(Box<CompiledExpr>),
    /// Logical NOT
    Not(Box<CompiledExpr>),
    /// Binary operation
    BinOp(BinOp, Box<CompiledExpr>, Box<CompiledExpr>),
    /// Function call: function name + argument list
    FuncCall(String, Vec<CompiledExpr>),
    /// Conditional IF(condition, then-branch, else-branch)
    If(Box<CompiledExpr>, Box<CompiledExpr>, Box<CompiledExpr>),
}

// ─── Errors ─────────────────────────────────────────────────────────────────

/// Error returned when an expression string cannot be parsed.
#[derive(Debug, Clone)]
pub struct CompileError(pub String);

impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CompileError: {}", self.0)
    }
}

impl std::error::Error for CompileError {}

/// Error returned when a compiled expression cannot be evaluated.
#[derive(Debug, Clone)]
pub struct EvalError(pub String);

impl fmt::Display for EvalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "EvalError: {}", self.0)
    }
}

impl std::error::Error for EvalError {}

// ─── Parser internals ────────────────────────────────────────────────────────

/// A simple hand-written recursive-descent parser for SPARQL filter expressions.
struct Parser<'a> {
    input: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn new(s: &'a str) -> Self {
        Self {
            input: s.as_bytes(),
            pos: 0,
        }
    }

    fn peek(&self) -> Option<u8> {
        self.input.get(self.pos).copied()
    }

    fn advance(&mut self) {
        if self.pos < self.input.len() {
            self.pos += 1;
        }
    }

    fn skip_ws(&mut self) {
        while let Some(c) = self.peek() {
            if c == b' ' || c == b'\t' || c == b'\r' || c == b'\n' {
                self.advance();
            } else {
                break;
            }
        }
    }

    fn expect(&mut self, ch: u8) -> Result<(), CompileError> {
        self.skip_ws();
        match self.peek() {
            Some(c) if c == ch => {
                self.advance();
                Ok(())
            }
            other => Err(CompileError(format!(
                "expected '{}' at pos {}, got {:?}",
                ch as char,
                self.pos,
                other.map(|b| b as char)
            ))),
        }
    }

    /// Parse a comma-separated list of expressions up to a closing paren.
    fn parse_arg_list(&mut self) -> Result<Vec<CompiledExpr>, CompileError> {
        self.expect(b'(')?;
        self.skip_ws();
        let mut args = Vec::new();
        if self.peek() == Some(b')') {
            self.advance();
            return Ok(args);
        }
        loop {
            args.push(self.parse_or()?);
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.advance();
                }
                Some(b')') => {
                    self.advance();
                    break;
                }
                other => {
                    return Err(CompileError(format!(
                        "expected ',' or ')' at pos {}, got {:?}",
                        self.pos,
                        other.map(|b| b as char)
                    )))
                }
            }
        }
        Ok(args)
    }

    // ── Grammar (ascending precedence) ──────────────────────────────────────
    // or  → and ( "||" and )*
    // and → cmp ( "&&" cmp )*
    // cmp → add ( ("="|"!="|"<"|"<="|">"|">=") add )?
    // add → mul ( ("+"|"-") mul )*
    // mul → unary ( ("*"|"/") unary )*
    // unary → "-" unary | "!" unary | primary
    // primary → literal | variable | iri | func | "(" or ")"

    fn parse_or(&mut self) -> Result<CompiledExpr, CompileError> {
        let mut left = self.parse_and()?;
        loop {
            self.skip_ws();
            if self.input.get(self.pos..self.pos + 2) == Some(b"||") {
                self.pos += 2;
                let right = self.parse_and()?;
                left = CompiledExpr::BinOp(BinOp::Or, Box::new(left), Box::new(right));
            } else {
                break;
            }
        }
        Ok(left)
    }

    fn parse_and(&mut self) -> Result<CompiledExpr, CompileError> {
        let mut left = self.parse_cmp()?;
        loop {
            self.skip_ws();
            if self.input.get(self.pos..self.pos + 2) == Some(b"&&") {
                self.pos += 2;
                let right = self.parse_cmp()?;
                left = CompiledExpr::BinOp(BinOp::And, Box::new(left), Box::new(right));
            } else {
                break;
            }
        }
        Ok(left)
    }

    fn parse_cmp(&mut self) -> Result<CompiledExpr, CompileError> {
        let left = self.parse_add()?;
        self.skip_ws();
        let op = if self.input.get(self.pos..self.pos + 2) == Some(b"!=") {
            self.pos += 2;
            Some(BinOp::Ne)
        } else if self.input.get(self.pos..self.pos + 2) == Some(b"<=") {
            self.pos += 2;
            Some(BinOp::Le)
        } else if self.input.get(self.pos..self.pos + 2) == Some(b">=") {
            self.pos += 2;
            Some(BinOp::Ge)
        } else if self.peek() == Some(b'=') {
            self.pos += 1;
            Some(BinOp::Eq)
        } else if self.peek() == Some(b'<') {
            self.pos += 1;
            Some(BinOp::Lt)
        } else if self.peek() == Some(b'>') {
            self.pos += 1;
            Some(BinOp::Gt)
        } else {
            None
        };
        if let Some(op) = op {
            let right = self.parse_add()?;
            Ok(CompiledExpr::BinOp(op, Box::new(left), Box::new(right)))
        } else {
            Ok(left)
        }
    }

    fn parse_add(&mut self) -> Result<CompiledExpr, CompileError> {
        let mut left = self.parse_mul()?;
        loop {
            self.skip_ws();
            match self.peek() {
                Some(b'+') => {
                    self.advance();
                    let right = self.parse_mul()?;
                    left = CompiledExpr::BinOp(BinOp::Add, Box::new(left), Box::new(right));
                }
                Some(b'-') => {
                    // Don't consume yet — could be a unary minus in next primary
                    // Consume only if it really is a binary minus (preceded by operand)
                    self.advance();
                    let right = self.parse_mul()?;
                    left = CompiledExpr::BinOp(BinOp::Sub, Box::new(left), Box::new(right));
                }
                _ => break,
            }
        }
        Ok(left)
    }

    fn parse_mul(&mut self) -> Result<CompiledExpr, CompileError> {
        let mut left = self.parse_unary()?;
        loop {
            self.skip_ws();
            match self.peek() {
                Some(b'*') => {
                    self.advance();
                    let right = self.parse_unary()?;
                    left = CompiledExpr::BinOp(BinOp::Mul, Box::new(left), Box::new(right));
                }
                Some(b'/') => {
                    self.advance();
                    let right = self.parse_unary()?;
                    left = CompiledExpr::BinOp(BinOp::Div, Box::new(left), Box::new(right));
                }
                _ => break,
            }
        }
        Ok(left)
    }

    fn parse_unary(&mut self) -> Result<CompiledExpr, CompileError> {
        self.skip_ws();
        match self.peek() {
            Some(b'-') => {
                self.advance();
                let inner = self.parse_unary()?;
                Ok(CompiledExpr::Neg(Box::new(inner)))
            }
            Some(b'!') => {
                self.advance();
                let inner = self.parse_unary()?;
                Ok(CompiledExpr::Not(Box::new(inner)))
            }
            _ => self.parse_primary(),
        }
    }

    fn parse_primary(&mut self) -> Result<CompiledExpr, CompileError> {
        self.skip_ws();
        match self.peek() {
            Some(b'"') => self.parse_string_literal(),
            Some(b'<') => self.parse_iri(),
            Some(b'?') => self.parse_variable(),
            Some(b'(') => {
                self.advance();
                let expr = self.parse_or()?;
                self.expect(b')')?;
                Ok(expr)
            }
            Some(c) if c.is_ascii_digit() => self.parse_number(),
            Some(c) if c.is_ascii_alphabetic() || c == b'_' => self.parse_name_or_call(),
            other => Err(CompileError(format!(
                "unexpected character at pos {}: {:?}",
                self.pos,
                other.map(|b| b as char)
            ))),
        }
    }

    fn parse_string_literal(&mut self) -> Result<CompiledExpr, CompileError> {
        self.advance(); // consume opening '"'
        let start = self.pos;
        while let Some(c) = self.peek() {
            if c == b'"' {
                break;
            }
            if c == b'\\' {
                self.advance();
            }
            self.advance();
        }
        let s = std::str::from_utf8(&self.input[start..self.pos])
            .map_err(|e| CompileError(format!("UTF-8 error in string: {e}")))?
            .to_string();
        self.expect(b'"')?;
        Ok(CompiledExpr::Literal(format!("\"{s}\"")))
    }

    fn parse_iri(&mut self) -> Result<CompiledExpr, CompileError> {
        self.advance(); // consume '<'
        let start = self.pos;
        while let Some(c) = self.peek() {
            if c == b'>' {
                break;
            }
            self.advance();
        }
        let iri = std::str::from_utf8(&self.input[start..self.pos])
            .map_err(|e| CompileError(format!("UTF-8 error in IRI: {e}")))?
            .to_string();
        self.expect(b'>')?;
        Ok(CompiledExpr::IriRef(iri))
    }

    fn parse_variable(&mut self) -> Result<CompiledExpr, CompileError> {
        self.advance(); // consume '?'
        let start = self.pos;
        while let Some(c) = self.peek() {
            if c.is_ascii_alphanumeric() || c == b'_' {
                self.advance();
            } else {
                break;
            }
        }
        let name = std::str::from_utf8(&self.input[start..self.pos])
            .map_err(|e| CompileError(format!("UTF-8 error in variable: {e}")))?
            .to_string();
        Ok(CompiledExpr::Variable(name))
    }

    fn parse_number(&mut self) -> Result<CompiledExpr, CompileError> {
        let start = self.pos;
        let mut has_dot = false;
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() {
                self.advance();
            } else if c == b'.' && !has_dot {
                has_dot = true;
                self.advance();
            } else {
                break;
            }
        }
        let num_str = std::str::from_utf8(&self.input[start..self.pos])
            .map_err(|e| CompileError(format!("UTF-8 error in number: {e}")))?;
        Ok(CompiledExpr::Literal(num_str.to_string()))
    }

    fn parse_name_or_call(&mut self) -> Result<CompiledExpr, CompileError> {
        let start = self.pos;
        while let Some(c) = self.peek() {
            if c.is_ascii_alphanumeric() || c == b'_' {
                self.advance();
            } else {
                break;
            }
        }
        let name = std::str::from_utf8(&self.input[start..self.pos])
            .map_err(|e| CompileError(format!("UTF-8 error in name: {e}")))?
            .to_string();

        self.skip_ws();
        // If followed by '(' it's a function call
        if self.peek() == Some(b'(') {
            let upper = name.to_uppercase();
            // Special-case IF
            if upper == "IF" {
                let args = self.parse_arg_list()?;
                if args.len() != 3 {
                    return Err(CompileError(format!(
                        "IF requires exactly 3 arguments, got {}",
                        args.len()
                    )));
                }
                let mut it = args.into_iter();
                let cond = it.next().expect("checked len");
                let then = it.next().expect("checked len");
                let else_ = it.next().expect("checked len");
                Ok(CompiledExpr::If(
                    Box::new(cond),
                    Box::new(then),
                    Box::new(else_),
                ))
            } else {
                let args = self.parse_arg_list()?;
                Ok(CompiledExpr::FuncCall(upper, args))
            }
        } else {
            // bare name — treat as a literal string token (e.g. "true"/"false")
            let lower = name.to_lowercase();
            if lower == "true" {
                Ok(CompiledExpr::Literal("true".to_string()))
            } else if lower == "false" {
                Ok(CompiledExpr::Literal("false".to_string()))
            } else {
                Err(CompileError(format!(
                    "unexpected bare identifier '{name}' at pos {start}"
                )))
            }
        }
    }
}

// ─── Compiler ────────────────────────────────────────────────────────────────

/// Compiles SPARQL filter expression strings into `CompiledExpr` trees.
#[derive(Debug, Clone, Default)]
pub struct ExprCompiler;

impl ExprCompiler {
    /// Create a new compiler instance.
    pub fn new() -> Self {
        Self
    }

    /// Compile an expression string into a `CompiledExpr`.
    pub fn compile(&self, expr_str: &str) -> Result<CompiledExpr, CompileError> {
        let mut p = Parser::new(expr_str.trim());
        let expr = p.parse_or()?;
        p.skip_ws();
        if p.pos != p.input.len() {
            return Err(CompileError(format!(
                "unexpected trailing input at pos {}: '{}'",
                p.pos,
                std::str::from_utf8(&p.input[p.pos..]).unwrap_or("<invalid utf8>")
            )));
        }
        Ok(expr)
    }

    /// Evaluate a compiled expression with the given variable bindings.
    pub fn evaluate(
        &self,
        expr: &CompiledExpr,
        bindings: &HashMap<String, ExprValue>,
    ) -> Result<ExprValue, EvalError> {
        match expr {
            CompiledExpr::Literal(s) => Self::eval_literal(s),
            CompiledExpr::Variable(name) => {
                Ok(bindings.get(name).cloned().unwrap_or(ExprValue::Unbound))
            }
            CompiledExpr::IriRef(iri) => Ok(ExprValue::Iri(iri.clone())),
            CompiledExpr::Neg(inner) => {
                let v = self.evaluate(inner, bindings)?;
                match v {
                    ExprValue::Integer(i) => Ok(ExprValue::Integer(-i)),
                    ExprValue::Double(d) => Ok(ExprValue::Double(-d)),
                    _ => Err(EvalError(format!("cannot negate {v}"))),
                }
            }
            CompiledExpr::Not(inner) => {
                let v = self.evaluate(inner, bindings)?;
                match v {
                    ExprValue::Bool(b) => Ok(ExprValue::Bool(!b)),
                    _ => Err(EvalError(format!("! requires boolean, got {v}"))),
                }
            }
            CompiledExpr::BinOp(op, left, right) => self.eval_binop(op, left, right, bindings),
            CompiledExpr::FuncCall(name, args) => self.eval_func(name, args, bindings),
            CompiledExpr::If(cond, then_expr, else_expr) => {
                let cv = self.evaluate(cond, bindings)?;
                match cv {
                    ExprValue::Bool(true) => self.evaluate(then_expr, bindings),
                    ExprValue::Bool(false) => self.evaluate(else_expr, bindings),
                    _ => Err(EvalError(format!("IF condition must be boolean, got {cv}"))),
                }
            }
        }
    }

    fn eval_literal(s: &str) -> Result<ExprValue, EvalError> {
        // Double-quoted string
        if s.starts_with('"') && s.ends_with('"') && s.len() >= 2 {
            return Ok(ExprValue::Str(s[1..s.len() - 1].to_string()));
        }
        // Boolean
        if s == "true" {
            return Ok(ExprValue::Bool(true));
        }
        if s == "false" {
            return Ok(ExprValue::Bool(false));
        }
        // Integer
        if let Ok(i) = s.parse::<i64>() {
            return Ok(ExprValue::Integer(i));
        }
        // Double
        if let Ok(d) = s.parse::<f64>() {
            return Ok(ExprValue::Double(d));
        }
        Err(EvalError(format!("cannot parse literal: '{s}'")))
    }

    fn eval_binop(
        &self,
        op: &BinOp,
        left: &CompiledExpr,
        right: &CompiledExpr,
        bindings: &HashMap<String, ExprValue>,
    ) -> Result<ExprValue, EvalError> {
        // Short-circuit evaluation for And/Or
        if *op == BinOp::And {
            let lv = self.evaluate(left, bindings)?;
            match lv {
                ExprValue::Bool(false) => return Ok(ExprValue::Bool(false)),
                ExprValue::Bool(true) => {
                    let rv = self.evaluate(right, bindings)?;
                    return match rv {
                        ExprValue::Bool(b) => Ok(ExprValue::Bool(b)),
                        _ => Err(EvalError(format!("&& requires boolean RHS, got {rv}"))),
                    };
                }
                _ => return Err(EvalError(format!("&& requires boolean LHS, got {lv}"))),
            }
        }
        if *op == BinOp::Or {
            let lv = self.evaluate(left, bindings)?;
            match lv {
                ExprValue::Bool(true) => return Ok(ExprValue::Bool(true)),
                ExprValue::Bool(false) => {
                    let rv = self.evaluate(right, bindings)?;
                    return match rv {
                        ExprValue::Bool(b) => Ok(ExprValue::Bool(b)),
                        _ => Err(EvalError(format!("|| requires boolean RHS, got {rv}"))),
                    };
                }
                _ => return Err(EvalError(format!("|| requires boolean LHS, got {lv}"))),
            }
        }

        let lv = self.evaluate(left, bindings)?;
        let rv = self.evaluate(right, bindings)?;

        match op {
            BinOp::Add => Self::numeric_op(&lv, &rv, |a, b| a + b, |a, b| a + b),
            BinOp::Sub => Self::numeric_op(&lv, &rv, |a, b| a - b, |a, b| a - b),
            BinOp::Mul => Self::numeric_op(&lv, &rv, |a, b| a * b, |a, b| a * b),
            BinOp::Div => match (&lv, &rv) {
                (ExprValue::Integer(_), ExprValue::Integer(0)) => {
                    Err(EvalError("division by zero".to_string()))
                }
                (ExprValue::Double(_), ExprValue::Double(d)) if *d == 0.0 => {
                    Err(EvalError("division by zero (double)".to_string()))
                }
                _ => Self::numeric_op(&lv, &rv, |a, b| a / b, |a, b| a / b),
            },
            BinOp::Eq => Ok(ExprValue::Bool(Self::values_equal(&lv, &rv))),
            BinOp::Ne => Ok(ExprValue::Bool(!Self::values_equal(&lv, &rv))),
            BinOp::Lt => Self::compare_op(&lv, &rv, std::cmp::Ordering::Less),
            BinOp::Le => Self::compare_op_le(&lv, &rv),
            BinOp::Gt => Self::compare_op(&lv, &rv, std::cmp::Ordering::Greater),
            BinOp::Ge => Self::compare_op_ge(&lv, &rv),
            BinOp::And | BinOp::Or => unreachable!("handled above"),
        }
    }

    fn numeric_op<FI, FD>(
        lv: &ExprValue,
        rv: &ExprValue,
        fi: FI,
        fd: FD,
    ) -> Result<ExprValue, EvalError>
    where
        FI: Fn(i64, i64) -> i64,
        FD: Fn(f64, f64) -> f64,
    {
        match (lv, rv) {
            (ExprValue::Integer(a), ExprValue::Integer(b)) => Ok(ExprValue::Integer(fi(*a, *b))),
            (ExprValue::Double(a), ExprValue::Double(b)) => Ok(ExprValue::Double(fd(*a, *b))),
            (ExprValue::Integer(a), ExprValue::Double(b)) => {
                Ok(ExprValue::Double(fd(*a as f64, *b)))
            }
            (ExprValue::Double(a), ExprValue::Integer(b)) => {
                Ok(ExprValue::Double(fd(*a, *b as f64)))
            }
            _ => Err(EvalError(format!(
                "numeric operation requires numeric operands, got {lv} and {rv}"
            ))),
        }
    }

    fn values_equal(a: &ExprValue, b: &ExprValue) -> bool {
        match (a, b) {
            (ExprValue::Bool(x), ExprValue::Bool(y)) => x == y,
            (ExprValue::Integer(x), ExprValue::Integer(y)) => x == y,
            (ExprValue::Double(x), ExprValue::Double(y)) => x == y,
            (ExprValue::Integer(x), ExprValue::Double(y)) => (*x as f64) == *y,
            (ExprValue::Double(x), ExprValue::Integer(y)) => *x == (*y as f64),
            (ExprValue::Str(x), ExprValue::Str(y)) => x == y,
            (ExprValue::Iri(x), ExprValue::Iri(y)) => x == y,
            (ExprValue::Blank(x), ExprValue::Blank(y)) => x == y,
            (ExprValue::Unbound, ExprValue::Unbound) => true,
            _ => false,
        }
    }

    fn compare_op(
        lv: &ExprValue,
        rv: &ExprValue,
        target: std::cmp::Ordering,
    ) -> Result<ExprValue, EvalError> {
        let ord = Self::numeric_cmp(lv, rv)?;
        Ok(ExprValue::Bool(ord == target))
    }

    fn compare_op_le(lv: &ExprValue, rv: &ExprValue) -> Result<ExprValue, EvalError> {
        let ord = Self::numeric_cmp(lv, rv)?;
        Ok(ExprValue::Bool(
            ord == std::cmp::Ordering::Less || ord == std::cmp::Ordering::Equal,
        ))
    }

    fn compare_op_ge(lv: &ExprValue, rv: &ExprValue) -> Result<ExprValue, EvalError> {
        let ord = Self::numeric_cmp(lv, rv)?;
        Ok(ExprValue::Bool(
            ord == std::cmp::Ordering::Greater || ord == std::cmp::Ordering::Equal,
        ))
    }

    fn numeric_cmp(lv: &ExprValue, rv: &ExprValue) -> Result<std::cmp::Ordering, EvalError> {
        match (lv, rv) {
            (ExprValue::Integer(a), ExprValue::Integer(b)) => Ok(a.cmp(b)),
            (ExprValue::Double(a), ExprValue::Double(b)) => a
                .partial_cmp(b)
                .ok_or_else(|| EvalError("NaN comparison".to_string())),
            (ExprValue::Integer(a), ExprValue::Double(b)) => (*a as f64)
                .partial_cmp(b)
                .ok_or_else(|| EvalError("NaN".to_string())),
            (ExprValue::Double(a), ExprValue::Integer(b)) => a
                .partial_cmp(&(*b as f64))
                .ok_or_else(|| EvalError("NaN".to_string())),
            (ExprValue::Str(a), ExprValue::Str(b)) => Ok(a.cmp(b)),
            _ => Err(EvalError(format!("cannot compare {lv} and {rv}"))),
        }
    }

    fn eval_func(
        &self,
        name: &str,
        args: &[CompiledExpr],
        bindings: &HashMap<String, ExprValue>,
    ) -> Result<ExprValue, EvalError> {
        match name {
            "BOUND" => {
                if args.len() != 1 {
                    return Err(EvalError(format!(
                        "BOUND expects 1 arg, got {}",
                        args.len()
                    )));
                }
                let v = self.evaluate(&args[0], bindings)?;
                Ok(ExprValue::Bool(!matches!(v, ExprValue::Unbound)))
            }
            "ISIRI" | "ISURI" => {
                if args.len() != 1 {
                    return Err(EvalError(format!(
                        "{name} expects 1 arg, got {}",
                        args.len()
                    )));
                }
                let v = self.evaluate(&args[0], bindings)?;
                Ok(ExprValue::Bool(matches!(v, ExprValue::Iri(_))))
            }
            "ISLITERAL" => {
                if args.len() != 1 {
                    return Err(EvalError(format!(
                        "ISLITERAL expects 1 arg, got {}",
                        args.len()
                    )));
                }
                let v = self.evaluate(&args[0], bindings)?;
                Ok(ExprValue::Bool(matches!(
                    v,
                    ExprValue::Str(_)
                        | ExprValue::Integer(_)
                        | ExprValue::Double(_)
                        | ExprValue::Bool(_)
                )))
            }
            "ISBLANK" => {
                if args.len() != 1 {
                    return Err(EvalError(format!(
                        "ISBLANK expects 1 arg, got {}",
                        args.len()
                    )));
                }
                let v = self.evaluate(&args[0], bindings)?;
                Ok(ExprValue::Bool(matches!(v, ExprValue::Blank(_))))
            }
            "STR" => {
                if args.len() != 1 {
                    return Err(EvalError(format!("STR expects 1 arg, got {}", args.len())));
                }
                let v = self.evaluate(&args[0], bindings)?;
                let s = match &v {
                    ExprValue::Str(s) => s.clone(),
                    ExprValue::Iri(s) => s.clone(),
                    ExprValue::Integer(i) => i.to_string(),
                    ExprValue::Double(d) => d.to_string(),
                    ExprValue::Bool(b) => b.to_string(),
                    ExprValue::Blank(b) => b.clone(),
                    ExprValue::Unbound => return Err(EvalError("STR of UNBOUND".to_string())),
                };
                Ok(ExprValue::Str(s))
            }
            "LANG" => {
                if args.len() != 1 {
                    return Err(EvalError(format!("LANG expects 1 arg, got {}", args.len())));
                }
                let v = self.evaluate(&args[0], bindings)?;
                match v {
                    ExprValue::Str(_) => Ok(ExprValue::Str(String::new())),
                    _ => Err(EvalError(format!("LANG requires string literal, got {v}"))),
                }
            }
            "DATATYPE" => {
                if args.len() != 1 {
                    return Err(EvalError(format!(
                        "DATATYPE expects 1 arg, got {}",
                        args.len()
                    )));
                }
                let v = self.evaluate(&args[0], bindings)?;
                let dt = match v {
                    ExprValue::Str(_) => "http://www.w3.org/2001/XMLSchema#string",
                    ExprValue::Integer(_) => "http://www.w3.org/2001/XMLSchema#integer",
                    ExprValue::Double(_) => "http://www.w3.org/2001/XMLSchema#double",
                    ExprValue::Bool(_) => "http://www.w3.org/2001/XMLSchema#boolean",
                    _ => return Err(EvalError("DATATYPE not applicable".to_string())),
                };
                Ok(ExprValue::Iri(dt.to_string()))
            }
            "COALESCE" => {
                for arg in args {
                    let v = self.evaluate(arg, bindings)?;
                    if !matches!(v, ExprValue::Unbound) {
                        return Ok(v);
                    }
                }
                Ok(ExprValue::Unbound)
            }
            "IF" => {
                // IF as a function call (alternative to CompiledExpr::If)
                if args.len() != 3 {
                    return Err(EvalError(format!("IF expects 3 args, got {}", args.len())));
                }
                let cv = self.evaluate(&args[0], bindings)?;
                match cv {
                    ExprValue::Bool(true) => self.evaluate(&args[1], bindings),
                    ExprValue::Bool(false) => self.evaluate(&args[2], bindings),
                    _ => Err(EvalError(format!("IF condition must be boolean, got {cv}"))),
                }
            }
            other => Err(EvalError(format!("unknown function: {other}"))),
        }
    }
}

// ─── LRU cache ──────────────────────────────────────────────────────────────

/// LRU expression cache backed by a `VecDeque` (order) + `HashMap` (storage).
///
/// Compiled expressions are stored in the map; the deque tracks access order
/// from oldest (front) to newest (back). On capacity overflow the front entry
/// is evicted.
pub struct ExprCache {
    cache: HashMap<String, CompiledExpr>,
    order: VecDeque<String>,
    max_size: usize,
    compiler: ExprCompiler,
    hits: u64,
    misses: u64,
}

impl ExprCache {
    /// Create a new cache with the given maximum number of entries.
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: HashMap::new(),
            order: VecDeque::new(),
            max_size,
            compiler: ExprCompiler::new(),
            hits: 0,
            misses: 0,
        }
    }

    /// Retrieve a compiled expression from cache, compiling it if not present.
    pub fn get_or_compile(&mut self, expr_str: &str) -> Result<&CompiledExpr, CompileError> {
        if self.cache.contains_key(expr_str) {
            // Move to back (most-recently-used)
            if let Some(pos) = self.order.iter().position(|k| k == expr_str) {
                self.order.remove(pos);
            }
            self.order.push_back(expr_str.to_string());
            self.hits += 1;
            return Ok(self.cache.get(expr_str).expect("just confirmed present"));
        }

        // Cache miss — compile
        self.misses += 1;
        let compiled = self.compiler.compile(expr_str)?;

        // Evict LRU entry if at capacity
        if self.cache.len() >= self.max_size && !self.cache.is_empty() {
            if let Some(oldest) = self.order.pop_front() {
                self.cache.remove(&oldest);
            }
        }

        self.cache.insert(expr_str.to_string(), compiled);
        self.order.push_back(expr_str.to_string());
        Ok(self.cache.get(expr_str).expect("just inserted"))
    }

    /// Current number of entries in the cache.
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }

    /// Clear all cached entries and reset statistics.
    pub fn clear(&mut self) {
        self.cache.clear();
        self.order.clear();
        self.hits = 0;
        self.misses = 0;
    }

    /// Number of cache hits since last clear.
    pub fn hits(&self) -> u64 {
        self.hits
    }

    /// Number of cache misses since last clear.
    pub fn misses(&self) -> u64 {
        self.misses
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn bindings(pairs: &[(&str, ExprValue)]) -> HashMap<String, ExprValue> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect()
    }

    fn compiler() -> ExprCompiler {
        ExprCompiler::new()
    }

    fn compile(s: &str) -> CompiledExpr {
        compiler().compile(s).expect("compile failed")
    }

    fn eval(expr: &CompiledExpr, b: &HashMap<String, ExprValue>) -> ExprValue {
        compiler().evaluate(expr, b).expect("eval failed")
    }

    // ── Literal parsing ─────────────────────────────────────────────────────

    #[test]
    fn test_literal_integer() {
        let e = compile("42");
        let v = eval(&e, &bindings(&[]));
        assert_eq!(v, ExprValue::Integer(42));
    }

    #[test]
    fn test_literal_zero() {
        let e = compile("0");
        let v = eval(&e, &bindings(&[]));
        assert_eq!(v, ExprValue::Integer(0));
    }

    #[test]
    fn test_literal_double() {
        let e = compile("3.15");
        let v = eval(&e, &bindings(&[]));
        assert_eq!(v, ExprValue::Double(3.15));
    }

    #[test]
    fn test_literal_string() {
        let e = compile(r#""hello world""#);
        let v = eval(&e, &bindings(&[]));
        assert_eq!(v, ExprValue::Str("hello world".to_string()));
    }

    #[test]
    fn test_literal_empty_string() {
        let e = compile(r#""""#);
        let v = eval(&e, &bindings(&[]));
        assert_eq!(v, ExprValue::Str(String::new()));
    }

    #[test]
    fn test_literal_true() {
        let e = compile("true");
        let v = eval(&e, &bindings(&[]));
        assert_eq!(v, ExprValue::Bool(true));
    }

    #[test]
    fn test_literal_false() {
        let e = compile("false");
        let v = eval(&e, &bindings(&[]));
        assert_eq!(v, ExprValue::Bool(false));
    }

    // ── Variable lookup ─────────────────────────────────────────────────────

    #[test]
    fn test_variable_bound() {
        let e = compile("?x");
        let v = eval(&e, &bindings(&[("x", ExprValue::Integer(7))]));
        assert_eq!(v, ExprValue::Integer(7));
    }

    #[test]
    fn test_variable_unbound() {
        let e = compile("?missing");
        let v = eval(&e, &bindings(&[]));
        assert_eq!(v, ExprValue::Unbound);
    }

    #[test]
    fn test_variable_string_value() {
        let e = compile("?name");
        let v = eval(
            &e,
            &bindings(&[("name", ExprValue::Str("Alice".to_string()))]),
        );
        assert_eq!(v, ExprValue::Str("Alice".to_string()));
    }

    // ── IRI ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_iri_ref() {
        let e = compile("<http://example.org/foo>");
        let v = eval(&e, &bindings(&[]));
        assert_eq!(v, ExprValue::Iri("http://example.org/foo".to_string()));
    }

    // ── Arithmetic ──────────────────────────────────────────────────────────

    #[test]
    fn test_add_integers() {
        let e = compile("1 + 2");
        let v = eval(&e, &bindings(&[]));
        assert_eq!(v, ExprValue::Integer(3));
    }

    #[test]
    fn test_sub_integers() {
        let e = compile("10 - 3");
        let v = eval(&e, &bindings(&[]));
        assert_eq!(v, ExprValue::Integer(7));
    }

    #[test]
    fn test_mul_integers() {
        let e = compile("4 * 5");
        let v = eval(&e, &bindings(&[]));
        assert_eq!(v, ExprValue::Integer(20));
    }

    #[test]
    fn test_div_integers() {
        let e = compile("10 / 2");
        let v = eval(&e, &bindings(&[]));
        assert_eq!(v, ExprValue::Integer(5));
    }

    #[test]
    fn test_add_doubles() {
        let e = compile("1.5 + 2.5");
        let v = eval(&e, &bindings(&[]));
        assert_eq!(v, ExprValue::Double(4.0));
    }

    #[test]
    fn test_arithmetic_with_variable() {
        let e = compile("?x + 10");
        let v = eval(&e, &bindings(&[("x", ExprValue::Integer(5))]));
        assert_eq!(v, ExprValue::Integer(15));
    }

    #[test]
    fn test_unary_neg_integer() {
        let e = compile("-5");
        let v = eval(&e, &bindings(&[]));
        assert_eq!(v, ExprValue::Integer(-5));
    }

    #[test]
    fn test_unary_neg_double() {
        let e = compile("-3.15");
        let v = eval(&e, &bindings(&[]));
        assert_eq!(v, ExprValue::Double(-3.15));
    }

    // ── Comparison ──────────────────────────────────────────────────────────

    #[test]
    fn test_eq_true() {
        let e = compile("5 = 5");
        let v = eval(&e, &bindings(&[]));
        assert_eq!(v, ExprValue::Bool(true));
    }

    #[test]
    fn test_eq_false() {
        let e = compile("5 = 6");
        let v = eval(&e, &bindings(&[]));
        assert_eq!(v, ExprValue::Bool(false));
    }

    #[test]
    fn test_ne() {
        let e = compile("5 != 6");
        let v = eval(&e, &bindings(&[]));
        assert_eq!(v, ExprValue::Bool(true));
    }

    #[test]
    fn test_lt() {
        let e = compile("3 < 5");
        let v = eval(&e, &bindings(&[]));
        assert_eq!(v, ExprValue::Bool(true));
    }

    #[test]
    fn test_le() {
        let e = compile("5 <= 5");
        let v = eval(&e, &bindings(&[]));
        assert_eq!(v, ExprValue::Bool(true));
    }

    #[test]
    fn test_gt() {
        let e = compile("7 > 5");
        let v = eval(&e, &bindings(&[]));
        assert_eq!(v, ExprValue::Bool(true));
    }

    #[test]
    fn test_ge() {
        let e = compile("5 >= 5");
        let v = eval(&e, &bindings(&[]));
        assert_eq!(v, ExprValue::Bool(true));
    }

    // ── Boolean ─────────────────────────────────────────────────────────────

    #[test]
    fn test_and_true() {
        let e = compile("true && true");
        let v = eval(&e, &bindings(&[]));
        assert_eq!(v, ExprValue::Bool(true));
    }

    #[test]
    fn test_and_false() {
        let e = compile("true && false");
        let v = eval(&e, &bindings(&[]));
        assert_eq!(v, ExprValue::Bool(false));
    }

    #[test]
    fn test_or_false_true() {
        let e = compile("false || true");
        let v = eval(&e, &bindings(&[]));
        assert_eq!(v, ExprValue::Bool(true));
    }

    #[test]
    fn test_not_false() {
        let e = compile("!false");
        let v = eval(&e, &bindings(&[]));
        assert_eq!(v, ExprValue::Bool(true));
    }

    #[test]
    fn test_not_true() {
        let e = compile("!true");
        let v = eval(&e, &bindings(&[]));
        assert_eq!(v, ExprValue::Bool(false));
    }

    // ── IF ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_if_true_branch() {
        let e = compile("IF(true, 1, 2)");
        let v = eval(&e, &bindings(&[]));
        assert_eq!(v, ExprValue::Integer(1));
    }

    #[test]
    fn test_if_false_branch() {
        let e = compile("IF(false, 1, 2)");
        let v = eval(&e, &bindings(&[]));
        assert_eq!(v, ExprValue::Integer(2));
    }

    #[test]
    fn test_if_with_condition_expr() {
        let e = compile("IF(3 > 2, 100, 200)");
        let v = eval(&e, &bindings(&[]));
        assert_eq!(v, ExprValue::Integer(100));
    }

    // ── Built-in functions ───────────────────────────────────────────────────

    #[test]
    fn test_bound_true() {
        let e = compile("BOUND(?x)");
        let v = eval(&e, &bindings(&[("x", ExprValue::Integer(1))]));
        assert_eq!(v, ExprValue::Bool(true));
    }

    #[test]
    fn test_bound_false() {
        let e = compile("BOUND(?missing)");
        let v = eval(&e, &bindings(&[]));
        assert_eq!(v, ExprValue::Bool(false));
    }

    #[test]
    fn test_isiri_true() {
        let e = compile("ISIRI(?x)");
        let v = eval(
            &e,
            &bindings(&[("x", ExprValue::Iri("http://ex.org/".to_string()))]),
        );
        assert_eq!(v, ExprValue::Bool(true));
    }

    #[test]
    fn test_isiri_false() {
        let e = compile("ISIRI(?x)");
        let v = eval(
            &e,
            &bindings(&[("x", ExprValue::Str("not-iri".to_string()))]),
        );
        assert_eq!(v, ExprValue::Bool(false));
    }

    #[test]
    fn test_isliteral_integer() {
        let e = compile("ISLITERAL(?x)");
        let v = eval(&e, &bindings(&[("x", ExprValue::Integer(42))]));
        assert_eq!(v, ExprValue::Bool(true));
    }

    #[test]
    fn test_isblank_true() {
        let e = compile("ISBLANK(?b)");
        let v = eval(&e, &bindings(&[("b", ExprValue::Blank("b1".to_string()))]));
        assert_eq!(v, ExprValue::Bool(true));
    }

    #[test]
    fn test_str_integer() {
        let e = compile("STR(?x)");
        let v = eval(&e, &bindings(&[("x", ExprValue::Integer(99))]));
        assert_eq!(v, ExprValue::Str("99".to_string()));
    }

    #[test]
    fn test_str_iri() {
        let e = compile("STR(?x)");
        let v = eval(
            &e,
            &bindings(&[("x", ExprValue::Iri("http://example.org/".to_string()))]),
        );
        assert_eq!(v, ExprValue::Str("http://example.org/".to_string()));
    }

    #[test]
    fn test_lang_plain_string() {
        let e = compile(r#"LANG(?x)"#);
        let v = eval(&e, &bindings(&[("x", ExprValue::Str("hello".to_string()))]));
        assert_eq!(v, ExprValue::Str(String::new()));
    }

    #[test]
    fn test_datatype_string() {
        let e = compile(r#"DATATYPE(?x)"#);
        let v = eval(&e, &bindings(&[("x", ExprValue::Str("hi".to_string()))]));
        assert_eq!(
            v,
            ExprValue::Iri("http://www.w3.org/2001/XMLSchema#string".to_string())
        );
    }

    #[test]
    fn test_datatype_integer() {
        let e = compile("DATATYPE(?x)");
        let v = eval(&e, &bindings(&[("x", ExprValue::Integer(1))]));
        assert_eq!(
            v,
            ExprValue::Iri("http://www.w3.org/2001/XMLSchema#integer".to_string())
        );
    }

    #[test]
    fn test_coalesce_first_bound() {
        let e = compile("COALESCE(?a, ?b, 42)");
        let v = eval(
            &e,
            &bindings(&[("a", ExprValue::Integer(1)), ("b", ExprValue::Integer(2))]),
        );
        assert_eq!(v, ExprValue::Integer(1));
    }

    #[test]
    fn test_coalesce_skip_unbound() {
        let e = compile("COALESCE(?missing, 99)");
        let v = eval(&e, &bindings(&[]));
        assert_eq!(v, ExprValue::Integer(99));
    }

    #[test]
    fn test_coalesce_all_unbound() {
        let e = compile("COALESCE(?a, ?b)");
        let v = eval(&e, &bindings(&[]));
        assert_eq!(v, ExprValue::Unbound);
    }

    // ── Nested expressions ────────────────────────────────────────────────────

    #[test]
    fn test_nested_arithmetic() {
        let e = compile("(2 + 3) * 4");
        let v = eval(&e, &bindings(&[]));
        assert_eq!(v, ExprValue::Integer(20));
    }

    #[test]
    fn test_nested_comparison_and() {
        let e = compile("3 > 2 && 5 < 10");
        let v = eval(&e, &bindings(&[]));
        assert_eq!(v, ExprValue::Bool(true));
    }

    #[test]
    fn test_complex_variable_expr() {
        let e = compile("?age >= 18 && ?age < 65");
        let v = eval(&e, &bindings(&[("age", ExprValue::Integer(30))]));
        assert_eq!(v, ExprValue::Bool(true));
    }

    // ── Error cases ───────────────────────────────────────────────────────────

    #[test]
    fn test_div_by_zero_error() {
        let e = compile("10 / 0");
        let result = compiler().evaluate(&e, &bindings(&[]));
        assert!(result.is_err());
    }

    #[test]
    fn test_type_mismatch_neg() {
        let e = compile(r#"-"hello""#);
        let result = compiler().evaluate(&e, &bindings(&[]));
        assert!(result.is_err());
    }

    #[test]
    fn test_compile_error_bare_name() {
        let result = ExprCompiler::new().compile("undefined_function");
        assert!(result.is_err());
    }

    #[test]
    fn test_if_wrong_arg_count() {
        let result = ExprCompiler::new().compile("IF(true, 1)");
        assert!(result.is_err());
    }

    // ── ExprCache ─────────────────────────────────────────────────────────────

    #[test]
    fn test_cache_basic() {
        let mut cache = ExprCache::new(10);
        cache.get_or_compile("1 + 1").expect("compile ok");
        assert_eq!(cache.cache_size(), 1);
        assert_eq!(cache.misses(), 1);
        assert_eq!(cache.hits(), 0);
    }

    #[test]
    fn test_cache_hit() {
        let mut cache = ExprCache::new(10);
        cache.get_or_compile("2 + 2").expect("ok");
        cache.get_or_compile("2 + 2").expect("ok");
        assert_eq!(cache.hits(), 1);
        assert_eq!(cache.misses(), 1);
        assert_eq!(cache.cache_size(), 1);
    }

    #[test]
    fn test_cache_lru_eviction() {
        let mut cache = ExprCache::new(3);
        cache.get_or_compile("1").expect("ok"); // miss, [1]
        cache.get_or_compile("2").expect("ok"); // miss, [1,2]
        cache.get_or_compile("3").expect("ok"); // miss, [1,2,3]
                                                // Access "1" again to make it recent
        cache.get_or_compile("1").expect("ok"); // hit, [2,3,1]
                                                // Insert "4" — should evict "2" (oldest)
        cache.get_or_compile("4").expect("ok"); // miss, [3,1,4]
        assert_eq!(cache.cache_size(), 3);
        // "2" should be evicted; "1" should still be there
        assert!(cache.cache.contains_key("1"));
        assert!(!cache.cache.contains_key("2"));
        assert!(cache.cache.contains_key("3"));
        assert!(cache.cache.contains_key("4"));
    }

    #[test]
    fn test_cache_clear() {
        let mut cache = ExprCache::new(5);
        cache.get_or_compile("42").expect("ok");
        cache.clear();
        assert_eq!(cache.cache_size(), 0);
        assert_eq!(cache.hits(), 0);
        assert_eq!(cache.misses(), 0);
    }

    #[test]
    fn test_cache_multiple_expressions() {
        let mut cache = ExprCache::new(10);
        let exprs = ["1 + 1", "2 * 3", "true && false", "?x = 5", "BOUND(?y)"];
        for s in &exprs {
            cache.get_or_compile(s).expect("ok");
        }
        assert_eq!(cache.cache_size(), exprs.len());
        assert_eq!(cache.misses(), exprs.len() as u64);
    }

    #[test]
    fn test_string_comparison() {
        let e = compile(r#""apple" = "apple""#);
        let v = eval(&e, &bindings(&[]));
        assert_eq!(v, ExprValue::Bool(true));
    }

    #[test]
    fn test_mixed_numeric_types() {
        let e = compile("3 + 1.5");
        let v = eval(&e, &bindings(&[]));
        assert_eq!(v, ExprValue::Double(4.5));
    }
}
