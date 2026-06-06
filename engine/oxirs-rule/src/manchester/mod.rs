//! OWL Manchester Syntax parser and emitter.
//!
//! This module provides a full implementation of the OWL 2 Manchester Syntax
//! as specified at <https://www.w3.org/TR/owl2-manchester-syntax/>.
//!
//! # Overview
//!
//! - [`ManchesterExpr`] — the AST for OWL class expressions
//! - [`ManchesterError`] — errors from lexing, parsing, or emitting
//! - [`parse`] — parse a Manchester Syntax string into a [`ManchesterExpr`]
//! - [`emit`] — serialize a [`ManchesterExpr`] back to Manchester Syntax text
//!
//! # Example
//!
//! ```rust
//! use oxirs_rule::manchester::{parse, emit, ManchesterExpr};
//!
//! let expr = parse("hasChild some Person").unwrap();
//! assert_eq!(
//!     expr,
//!     ManchesterExpr::Some {
//!         property: "hasChild".to_string(),
//!         filler: Box::new(ManchesterExpr::Class("Person".to_string())),
//!     }
//! );
//! let text = emit(&expr).unwrap();
//! assert_eq!(text, "hasChild some Person");
//! ```

/// OWL class expression in Manchester Syntax.
///
/// Each variant corresponds to a different OWL 2 class expression constructor
/// as defined in the Manchester Syntax specification.
#[derive(Debug, Clone, PartialEq)]
pub enum ManchesterExpr {
    /// Named class: `ClassName` or `prefix:LocalName`
    Class(String),
    /// ObjectIntersectionOf: `A and B` (binary or n-ary)
    And(Vec<ManchesterExpr>),
    /// ObjectUnionOf: `A or B` (binary or n-ary)
    Or(Vec<ManchesterExpr>),
    /// ObjectComplementOf: `not A`
    Not(Box<ManchesterExpr>),
    /// ObjectSomeValuesFrom: `P some C`
    Some {
        property: String,
        filler: Box<ManchesterExpr>,
    },
    /// ObjectAllValuesFrom: `P only C`
    Only {
        property: String,
        filler: Box<ManchesterExpr>,
    },
    /// ObjectMinCardinality: `P min N C` (filler is optional)
    Min {
        property: String,
        cardinality: u32,
        filler: Option<Box<ManchesterExpr>>,
    },
    /// ObjectMaxCardinality: `P max N C` (filler is optional)
    Max {
        property: String,
        cardinality: u32,
        filler: Option<Box<ManchesterExpr>>,
    },
    /// ObjectExactCardinality: `P exactly N C` (filler is optional)
    Exactly {
        property: String,
        cardinality: u32,
        filler: Option<Box<ManchesterExpr>>,
    },
    /// ObjectHasValue: `P value v`
    HasValue {
        property: String,
        individual: String,
    },
    /// ObjectOneOf: `{a b c}` — enumeration of individuals
    OneOf(Vec<String>),
}

/// Errors produced by the Manchester Syntax lexer, parser, or emitter.
#[derive(Debug, Clone, thiserror::Error)]
pub enum ManchesterError {
    /// Error from the tokenizer; `pos` is the byte offset in the input string.
    #[error("Lexer error at byte {pos}: {msg}")]
    LexError { pos: usize, msg: String },

    /// Error from the recursive-descent parser; `pos` is the token index.
    #[error("Parse error at token {pos}: {msg}")]
    ParseError { pos: usize, msg: String },

    /// Error from the AST serializer.
    #[error("Emit error: {0}")]
    EmitError(String),
}

pub mod emitter;
pub mod lexer;
pub mod parser;
#[cfg(test)]
mod tests;

pub use emitter::emit;
pub use parser::parse;
