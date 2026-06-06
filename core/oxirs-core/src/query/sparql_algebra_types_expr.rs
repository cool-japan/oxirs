//! SPARQL value expression algebra nodes.
//!
//! Contains [`Expression`], [`FunctionExpression`], [`BuiltInFunction`], and the
//! internal SSE formatting helpers shared with the algebraic-operations module.

use super::sparql_algebra_types_pattern::GraphPattern;
use crate::model::*;
use std::fmt;

// ────────────────────────────────────────────────────────────────────────────
// Expression
// ────────────────────────────────────────────────────────────────────────────

/// An [expression](https://www.w3.org/TR/sparql11-query/#expressions).
#[derive(Eq, PartialEq, Debug, Clone, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Expression {
    NamedNode(NamedNode),
    Literal(Literal),
    Variable(Variable),
    /// [Logical-or](https://www.w3.org/TR/sparql11-query/#func-logical-or).
    Or(Box<Self>, Box<Self>),
    /// [Logical-and](https://www.w3.org/TR/sparql11-query/#func-logical-and).
    And(Box<Self>, Box<Self>),
    /// [RDFterm-equal](https://www.w3.org/TR/sparql11-query/#func-RDFterm-equal) and all the XSD equalities.
    Equal(Box<Self>, Box<Self>),
    /// [sameTerm](https://www.w3.org/TR/sparql11-query/#func-sameTerm).
    SameTerm(Box<Self>, Box<Self>),
    /// [op:numeric-greater-than](https://www.w3.org/TR/xpath-functions-31/#func-numeric-greater-than) and other XSD greater than operators.
    Greater(Box<Self>, Box<Self>),
    GreaterOrEqual(Box<Self>, Box<Self>),
    /// [op:numeric-less-than](https://www.w3.org/TR/xpath-functions-31/#func-numeric-less-than) and other XSD greater than operators.
    Less(Box<Self>, Box<Self>),
    LessOrEqual(Box<Self>, Box<Self>),
    /// [IN](https://www.w3.org/TR/sparql11-query/#func-in)
    In(Box<Self>, Vec<Self>),
    /// [op:numeric-add](https://www.w3.org/TR/xpath-functions-31/#func-numeric-add) and other XSD additions.
    Add(Box<Self>, Box<Self>),
    /// [op:numeric-subtract](https://www.w3.org/TR/xpath-functions-31/#func-numeric-subtract) and other XSD subtractions.
    Subtract(Box<Self>, Box<Self>),
    /// [op:numeric-multiply](https://www.w3.org/TR/xpath-functions-31/#func-numeric-multiply) and other XSD multiplications.
    Multiply(Box<Self>, Box<Self>),
    /// [op:numeric-divide](https://www.w3.org/TR/xpath-functions-31/#func-numeric-divide) and other XSD divides.
    Divide(Box<Self>, Box<Self>),
    /// [op:numeric-unary-plus](https://www.w3.org/TR/xpath-functions-31/#func-numeric-unary-plus) and other XSD unary plus.
    UnaryPlus(Box<Self>),
    /// [op:numeric-unary-minus](https://www.w3.org/TR/xpath-functions-31/#func-numeric-unary-minus) and other XSD unary minus.
    UnaryMinus(Box<Self>),
    /// [fn:not](https://www.w3.org/TR/xpath-functions-31/#func-not).
    Not(Box<Self>),
    /// [EXISTS](https://www.w3.org/TR/sparql11-query/#func-filter-exists).
    Exists(Box<GraphPattern>),
    /// [BOUND](https://www.w3.org/TR/sparql11-query/#func-bound).
    Bound(Variable),
    /// [IF](https://www.w3.org/TR/sparql11-query/#func-if).
    If(Box<Self>, Box<Self>, Box<Self>),
    /// [COALESCE](https://www.w3.org/TR/sparql11-query/#func-coalesce).
    Coalesce(Vec<Self>),
    /// A regular function call.
    FunctionCall(FunctionExpression, Vec<Self>),
}

impl Expression {
    /// Formats using the SPARQL S-Expression syntax
    pub fn fmt_sse(&self, f: &mut impl fmt::Write) -> fmt::Result {
        match self {
            Self::NamedNode(node) => write!(f, "{node}"),
            Self::Literal(l) => write!(f, "{l}"),
            Self::Variable(var) => write!(f, "{var}"),
            Self::Or(a, b) => fmt_sse_binary_expression(f, "||", a, b),
            Self::And(a, b) => fmt_sse_binary_expression(f, "&&", a, b),
            Self::Equal(a, b) => fmt_sse_binary_expression(f, "=", a, b),
            Self::SameTerm(a, b) => fmt_sse_binary_expression(f, "sameTerm", a, b),
            Self::Greater(a, b) => fmt_sse_binary_expression(f, ">", a, b),
            Self::GreaterOrEqual(a, b) => fmt_sse_binary_expression(f, ">=", a, b),
            Self::Less(a, b) => fmt_sse_binary_expression(f, "<", a, b),
            Self::LessOrEqual(a, b) => fmt_sse_binary_expression(f, "<=", a, b),
            Self::In(a, b) => {
                f.write_str("(in ")?;
                a.fmt_sse(f)?;
                for p in b {
                    f.write_str(" ")?;
                    p.fmt_sse(f)?;
                }
                f.write_str(")")
            }
            Self::Add(a, b) => fmt_sse_binary_expression(f, "+", a, b),
            Self::Subtract(a, b) => fmt_sse_binary_expression(f, "-", a, b),
            Self::Multiply(a, b) => fmt_sse_binary_expression(f, "*", a, b),
            Self::Divide(a, b) => fmt_sse_binary_expression(f, "/", a, b),
            Self::UnaryPlus(e) => fmt_sse_unary_expression(f, "+", e),
            Self::UnaryMinus(e) => fmt_sse_unary_expression(f, "-", e),
            Self::Not(e) => fmt_sse_unary_expression(f, "!", e),
            Self::FunctionCall(function, parameters) => {
                f.write_str("( ")?;
                function.fmt_sse(f)?;
                for p in parameters {
                    f.write_str(" ")?;
                    p.fmt_sse(f)?;
                }
                f.write_str(")")
            }
            Self::Exists(p) => {
                f.write_str("(exists ")?;
                p.fmt_sse(f)?;
                f.write_str(")")
            }
            Self::Bound(v) => {
                write!(f, "(bound {v})")
            }
            Self::If(a, b, c) => {
                f.write_str("(if ")?;
                a.fmt_sse(f)?;
                f.write_str(" ")?;
                b.fmt_sse(f)?;
                f.write_str(" ")?;
                c.fmt_sse(f)?;
                f.write_str(")")
            }
            Self::Coalesce(parameters) => {
                f.write_str("(coalesce")?;
                for p in parameters {
                    f.write_str(" ")?;
                    p.fmt_sse(f)?;
                }
                f.write_str(")")
            }
        }
    }
}

impl fmt::Display for Expression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NamedNode(node) => node.fmt(f),
            Self::Literal(literal) => literal.fmt(f),
            Self::Variable(var) => var.fmt(f),
            Self::Or(left, right) => write!(f, "({left} || {right})"),
            Self::And(left, right) => write!(f, "({left} && {right})"),
            Self::Equal(left, right) => write!(f, "({left} = {right})"),
            Self::SameTerm(left, right) => write!(f, "sameTerm({left}, {right})"),
            Self::Greater(left, right) => write!(f, "({left} > {right})"),
            Self::GreaterOrEqual(left, right) => write!(f, "({left} >= {right})"),
            Self::Less(left, right) => write!(f, "({left} < {right})"),
            Self::LessOrEqual(left, right) => write!(f, "({left} <= {right})"),
            Self::In(expr, list) => {
                write!(f, "({expr} IN (")?;
                for (i, item) in list.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{item}")?;
                }
                f.write_str("))")
            }
            Self::Add(left, right) => write!(f, "({left} + {right})"),
            Self::Subtract(left, right) => write!(f, "({left} - {right})"),
            Self::Multiply(left, right) => write!(f, "({left} * {right})"),
            Self::Divide(left, right) => write!(f, "({left} / {right})"),
            Self::UnaryPlus(expr) => write!(f, "(+{expr})"),
            Self::UnaryMinus(expr) => write!(f, "(-{expr})"),
            Self::Not(expr) => write!(f, "(!{expr})"),
            Self::Exists(pattern) => write!(f, "EXISTS {{ {pattern} }}"),
            Self::Bound(var) => write!(f, "BOUND({var})"),
            Self::If(condition, then_expr, else_expr) => {
                write!(f, "IF({condition}, {then_expr}, {else_expr})")
            }
            Self::Coalesce(exprs) => {
                f.write_str("COALESCE(")?;
                for (i, expr) in exprs.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{expr}")?;
                }
                f.write_str(")")
            }
            Self::FunctionCall(func, args) => {
                write!(f, "{func}(")?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{arg}")?;
                }
                f.write_str(")")
            }
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// FunctionExpression
// ────────────────────────────────────────────────────────────────────────────

/// A function call
#[derive(Eq, PartialEq, Debug, Clone, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum FunctionExpression {
    /// A call to a built-in function
    BuiltIn(BuiltInFunction),
    /// A call to a custom function identified by an IRI
    Custom(NamedNode),
}

impl FunctionExpression {
    pub fn fmt_sse(&self, f: &mut impl fmt::Write) -> fmt::Result {
        match self {
            Self::BuiltIn(function) => function.fmt_sse(f),
            Self::Custom(iri) => write!(f, "{iri}"),
        }
    }
}

impl fmt::Display for FunctionExpression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BuiltIn(function) => function.fmt(f),
            Self::Custom(iri) => iri.fmt(f),
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// BuiltInFunction
// ────────────────────────────────────────────────────────────────────────────

/// Built-in SPARQL functions
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BuiltInFunction {
    // String functions
    Str,
    Lang,
    LangMatches,
    Datatype,
    Iri,
    Uri,
    Bnode,
    StrDt,
    StrLang,
    StrLen,
    SubStr,
    UCase,
    LCase,
    StrStarts,
    StrEnds,
    Contains,
    StrBefore,
    StrAfter,
    Encode,
    Concat,
    Replace,
    Regex,

    // Numeric functions
    Abs,
    Round,
    Ceil,
    Floor,
    Rand,

    // Date/Time functions
    Now,
    Year,
    Month,
    Day,
    Hours,
    Minutes,
    Seconds,
    Timezone,
    Tz,

    // Hash functions
    Md5,
    Sha1,
    Sha256,
    Sha384,
    Sha512,

    // Type checking
    IsIri,
    IsUri,
    IsBlank,
    IsLiteral,
    IsNumeric,

    // Additional functions
    Uuid,
    StrUuid,
}

impl BuiltInFunction {
    pub fn fmt_sse(&self, f: &mut impl fmt::Write) -> fmt::Result {
        match self {
            Self::Str => f.write_str("str"),
            Self::Lang => f.write_str("lang"),
            Self::LangMatches => f.write_str("langMatches"),
            Self::Datatype => f.write_str("datatype"),
            Self::Iri => f.write_str("iri"),
            Self::Uri => f.write_str("uri"),
            Self::Bnode => f.write_str("bnode"),
            Self::StrDt => f.write_str("strdt"),
            Self::StrLang => f.write_str("strlang"),
            Self::StrLen => f.write_str("strlen"),
            Self::SubStr => f.write_str("substr"),
            Self::UCase => f.write_str("ucase"),
            Self::LCase => f.write_str("lcase"),
            Self::StrStarts => f.write_str("strstarts"),
            Self::StrEnds => f.write_str("strends"),
            Self::Contains => f.write_str("contains"),
            Self::StrBefore => f.write_str("strbefore"),
            Self::StrAfter => f.write_str("strafter"),
            Self::Encode => f.write_str("encode_for_uri"),
            Self::Concat => f.write_str("concat"),
            Self::Replace => f.write_str("replace"),
            Self::Regex => f.write_str("regex"),
            Self::Abs => f.write_str("abs"),
            Self::Round => f.write_str("round"),
            Self::Ceil => f.write_str("ceil"),
            Self::Floor => f.write_str("floor"),
            Self::Rand => f.write_str("rand"),
            Self::Now => f.write_str("now"),
            Self::Year => f.write_str("year"),
            Self::Month => f.write_str("month"),
            Self::Day => f.write_str("day"),
            Self::Hours => f.write_str("hours"),
            Self::Minutes => f.write_str("minutes"),
            Self::Seconds => f.write_str("seconds"),
            Self::Timezone => f.write_str("timezone"),
            Self::Tz => f.write_str("tz"),
            Self::Md5 => f.write_str("md5"),
            Self::Sha1 => f.write_str("sha1"),
            Self::Sha256 => f.write_str("sha256"),
            Self::Sha384 => f.write_str("sha384"),
            Self::Sha512 => f.write_str("sha512"),
            Self::IsIri => f.write_str("isiri"),
            Self::IsUri => f.write_str("isuri"),
            Self::IsBlank => f.write_str("isblank"),
            Self::IsLiteral => f.write_str("isliteral"),
            Self::IsNumeric => f.write_str("isnumeric"),
            Self::Uuid => f.write_str("uuid"),
            Self::StrUuid => f.write_str("struuid"),
        }
    }
}

impl fmt::Display for BuiltInFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Str => f.write_str("STR"),
            Self::Lang => f.write_str("LANG"),
            Self::LangMatches => f.write_str("LANGMATCHES"),
            Self::Datatype => f.write_str("DATATYPE"),
            Self::Iri => f.write_str("IRI"),
            Self::Uri => f.write_str("URI"),
            Self::Bnode => f.write_str("BNODE"),
            Self::StrDt => f.write_str("STRDT"),
            Self::StrLang => f.write_str("STRLANG"),
            Self::StrLen => f.write_str("STRLEN"),
            Self::SubStr => f.write_str("SUBSTR"),
            Self::UCase => f.write_str("UCASE"),
            Self::LCase => f.write_str("LCASE"),
            Self::StrStarts => f.write_str("STRSTARTS"),
            Self::StrEnds => f.write_str("STRENDS"),
            Self::Contains => f.write_str("CONTAINS"),
            Self::StrBefore => f.write_str("STRBEFORE"),
            Self::StrAfter => f.write_str("STRAFTER"),
            Self::Encode => f.write_str("ENCODE_FOR_URI"),
            Self::Concat => f.write_str("CONCAT"),
            Self::Replace => f.write_str("REPLACE"),
            Self::Regex => f.write_str("REGEX"),
            Self::Abs => f.write_str("ABS"),
            Self::Round => f.write_str("ROUND"),
            Self::Ceil => f.write_str("CEIL"),
            Self::Floor => f.write_str("FLOOR"),
            Self::Rand => f.write_str("RAND"),
            Self::Now => f.write_str("NOW"),
            Self::Year => f.write_str("YEAR"),
            Self::Month => f.write_str("MONTH"),
            Self::Day => f.write_str("DAY"),
            Self::Hours => f.write_str("HOURS"),
            Self::Minutes => f.write_str("MINUTES"),
            Self::Seconds => f.write_str("SECONDS"),
            Self::Timezone => f.write_str("TIMEZONE"),
            Self::Tz => f.write_str("TZ"),
            Self::Md5 => f.write_str("MD5"),
            Self::Sha1 => f.write_str("SHA1"),
            Self::Sha256 => f.write_str("SHA256"),
            Self::Sha384 => f.write_str("SHA384"),
            Self::Sha512 => f.write_str("SHA512"),
            Self::IsIri => f.write_str("isIRI"),
            Self::IsUri => f.write_str("isURI"),
            Self::IsBlank => f.write_str("isBLANK"),
            Self::IsLiteral => f.write_str("isLITERAL"),
            Self::IsNumeric => f.write_str("isNUMERIC"),
            Self::Uuid => f.write_str("UUID"),
            Self::StrUuid => f.write_str("STRUUID"),
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Internal SSE formatting helpers (pub(crate) so ops module can reuse)
// ────────────────────────────────────────────────────────────────────────────

pub(crate) fn fmt_sse_binary_expression(
    f: &mut impl fmt::Write,
    operator: &str,
    left: &Expression,
    right: &Expression,
) -> fmt::Result {
    f.write_str("(")?;
    f.write_str(operator)?;
    f.write_str(" ")?;
    left.fmt_sse(f)?;
    f.write_str(" ")?;
    right.fmt_sse(f)?;
    f.write_str(")")
}

pub(crate) fn fmt_sse_unary_expression(
    f: &mut impl fmt::Write,
    operator: &str,
    expr: &Expression,
) -> fmt::Result {
    f.write_str("(")?;
    f.write_str(operator)?;
    f.write_str(" ")?;
    expr.fmt_sse(f)?;
    f.write_str(")")
}
