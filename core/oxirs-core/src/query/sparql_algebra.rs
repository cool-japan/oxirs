//! Enhanced SPARQL 1.1+ Query Algebra implementation
//!
//! Extracted and adapted from OxiGraph spargebra with OxiRS enhancements.
//! Based on W3C SPARQL 1.1 Query specification:
//! https://www.w3.org/TR/sparql11-query/#sparqlQuery

use crate::model::*;
use std::fmt;

/// A [property path expression](https://www.w3.org/TR/sparql11-query/#defn_PropertyPathExpr).
#[derive(Eq, PartialEq, Debug, Clone, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PropertyPathExpression {
    /// Simple named node predicate
    NamedNode(NamedNode),
    /// Inverse path: ^path
    Reverse(Box<Self>),
    /// Sequence path: path1 / path2
    Sequence(Box<Self>, Box<Self>),
    /// Alternative path: path1 | path2
    Alternative(Box<Self>, Box<Self>),
    /// Zero or more: path*
    ZeroOrMore(Box<Self>),
    /// One or more: path+
    OneOrMore(Box<Self>),
    /// Zero or one: path?
    ZeroOrOne(Box<Self>),
    /// Negated property set: !(p1 | p2 | ...)
    NegatedPropertySet(Vec<NamedNode>),
}

impl PropertyPathExpression {
    /// Formats using the SPARQL S-Expression syntax
    pub fn fmt_sse(&self, f: &mut impl fmt::Write) -> fmt::Result {
        match self {
            Self::NamedNode(p) => write!(f, "{p}"),
            Self::Reverse(p) => {
                f.write_str("(reverse ")?;
                p.fmt_sse(f)?;
                f.write_str(")")
            }
            Self::Alternative(a, b) => {
                f.write_str("(alt ")?;
                a.fmt_sse(f)?;
                f.write_str(" ")?;
                b.fmt_sse(f)?;
                f.write_str(")")
            }
            Self::Sequence(a, b) => {
                f.write_str("(seq ")?;
                a.fmt_sse(f)?;
                f.write_str(" ")?;
                b.fmt_sse(f)?;
                f.write_str(")")
            }
            Self::ZeroOrMore(p) => {
                f.write_str("(path* ")?;
                p.fmt_sse(f)?;
                f.write_str(")")
            }
            Self::OneOrMore(p) => {
                f.write_str("(path+ ")?;
                p.fmt_sse(f)?;
                f.write_str(")")
            }
            Self::ZeroOrOne(p) => {
                f.write_str("(path? ")?;
                p.fmt_sse(f)?;
                f.write_str(")")
            }
            Self::NegatedPropertySet(p) => {
                f.write_str("(notoneof")?;
                for p in p {
                    write!(f, " {p}")?;
                }
                f.write_str(")")
            }
        }
    }
}

impl fmt::Display for PropertyPathExpression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NamedNode(p) => p.fmt(f),
            Self::Reverse(p) => write!(f, "^({p})"),
            Self::Sequence(a, b) => write!(f, "({a} / {b})"),
            Self::Alternative(a, b) => write!(f, "({a} | {b})"),
            Self::ZeroOrMore(p) => write!(f, "({p})*"),
            Self::OneOrMore(p) => write!(f, "({p})+"),
            Self::ZeroOrOne(p) => write!(f, "({p})?"),
            Self::NegatedPropertySet(p) => {
                f.write_str("!(")?;
                for (i, c) in p.iter().enumerate() {
                    if i > 0 {
                        f.write_str(" | ")?;
                    }
                    write!(f, "{c}")?;
                }
                f.write_str(")")
            }
        }
    }
}

impl From<NamedNode> for PropertyPathExpression {
    fn from(p: NamedNode) -> Self {
        Self::NamedNode(p)
    }
}

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

/// A [SPARQL graph pattern](https://www.w3.org/TR/sparql11-query/#GraphPattern)
#[derive(Eq, PartialEq, Debug, Clone, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum GraphPattern {
    /// A [basic graph pattern](https://www.w3.org/TR/sparql11-query/#defn_BasicGraphPattern).
    Bgp { patterns: Vec<TriplePattern> },
    /// A [property path pattern](https://www.w3.org/TR/sparql11-query/#defn_evalPP_predicate).
    Path {
        subject: TermPattern,
        path: PropertyPathExpression,
        object: TermPattern,
    },
    /// [Join](https://www.w3.org/TR/sparql11-query/#defn_algJoin).
    Join { left: Box<Self>, right: Box<Self> },
    /// [LeftJoin](https://www.w3.org/TR/sparql11-query/#defn_algLeftJoin).
    LeftJoin {
        left: Box<Self>,
        right: Box<Self>,
        expression: Option<Expression>,
    },
    /// [Filter](https://www.w3.org/TR/sparql11-query/#defn_algFilter).
    Filter { expr: Expression, inner: Box<Self> },
    /// [Union](https://www.w3.org/TR/sparql11-query/#defn_algUnion).
    Union { left: Box<Self>, right: Box<Self> },
    /// Graph pattern (GRAPH clause)
    Graph {
        name: NamedNodePattern,
        inner: Box<Self>,
    },
    /// [Extend](https://www.w3.org/TR/sparql11-query/#defn_extend).
    Extend {
        inner: Box<Self>,
        variable: Variable,
        expression: Expression,
    },
    /// [Minus](https://www.w3.org/TR/sparql11-query/#defn_algMinus).
    Minus { left: Box<Self>, right: Box<Self> },
    /// A table used to provide inline values
    Values {
        variables: Vec<Variable>,
        bindings: Vec<Vec<Option<GroundTerm>>>,
    },
    /// [OrderBy](https://www.w3.org/TR/sparql11-query/#defn_algOrdered).
    OrderBy {
        inner: Box<Self>,
        expression: Vec<OrderExpression>,
    },
    /// [Project](https://www.w3.org/TR/sparql11-query/#defn_algProjection).
    Project {
        inner: Box<Self>,
        variables: Vec<Variable>,
    },
    /// [Distinct](https://www.w3.org/TR/sparql11-query/#defn_algDistinct).
    Distinct { inner: Box<Self> },
    /// [Reduced](https://www.w3.org/TR/sparql11-query/#defn_algReduced).
    Reduced { inner: Box<Self> },
    /// [Slice](https://www.w3.org/TR/sparql11-query/#defn_algSlice).
    Slice {
        inner: Box<Self>,
        start: usize,
        length: Option<usize>,
    },
    /// [Group](https://www.w3.org/TR/sparql11-query/#aggregateAlgebra).
    Group {
        inner: Box<Self>,
        variables: Vec<Variable>,
        aggregates: Vec<(Variable, AggregateExpression)>,
    },
    /// [Service](https://www.w3.org/TR/sparql11-federated-query/#defn_evalService).
    Service {
        name: NamedNodePattern,
        inner: Box<Self>,
        silent: bool,
    },
}

impl fmt::Display for GraphPattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bgp { patterns } => {
                for (i, pattern) in patterns.iter().enumerate() {
                    if i > 0 {
                        f.write_str(" . ")?;
                    }
                    write!(f, "{pattern}")?;
                }
                Ok(())
            }
            Self::Path {
                subject,
                path,
                object,
            } => {
                write!(f, "{subject} {path} {object}")
            }
            Self::Join { left, right } => {
                write!(f, "{left} . {right}")
            }
            Self::LeftJoin {
                left,
                right,
                expression,
            } => {
                write!(f, "{left} OPTIONAL {{ {right}")?;
                if let Some(expr) = expression {
                    write!(f, " FILTER ({expr})")?;
                }
                f.write_str(" }")
            }
            Self::Filter { expr, inner } => {
                write!(f, "{inner} FILTER ({expr})")
            }
            Self::Union { left, right } => {
                write!(f, "{{ {left} }} UNION {{ {right} }}")
            }
            Self::Graph { name, inner } => {
                write!(f, "GRAPH {name} {{ {inner} }}")
            }
            Self::Extend {
                inner,
                variable,
                expression,
            } => {
                write!(f, "{inner} BIND ({expression} AS {variable})")
            }
            Self::Minus { left, right } => {
                write!(f, "{left} MINUS {{ {right} }}")
            }
            Self::Values {
                variables,
                bindings,
            } => {
                f.write_str("VALUES ")?;
                if variables.len() == 1 {
                    write!(f, "{}", variables[0])?;
                } else {
                    f.write_str("(")?;
                    for (i, var) in variables.iter().enumerate() {
                        if i > 0 {
                            f.write_str(" ")?;
                        }
                        write!(f, "{var}")?;
                    }
                    f.write_str(")")?;
                }
                f.write_str(" { ")?;
                for (i, binding) in bindings.iter().enumerate() {
                    if i > 0 {
                        f.write_str(" ")?;
                    }
                    if variables.len() == 1 {
                        if let Some(term) = &binding[0] {
                            write!(f, "{term}")?;
                        } else {
                            f.write_str("UNDEF")?;
                        }
                    } else {
                        f.write_str("(")?;
                        for (j, value) in binding.iter().enumerate() {
                            if j > 0 {
                                f.write_str(" ")?;
                            }
                            if let Some(term) = value {
                                write!(f, "{term}")?;
                            } else {
                                f.write_str("UNDEF")?;
                            }
                        }
                        f.write_str(")")?;
                    }
                }
                f.write_str(" }")
            }
            Self::OrderBy { inner, expression } => {
                write!(f, "{inner} ORDER BY")?;
                for expr in expression {
                    write!(f, " {expr}")?;
                }
                Ok(())
            }
            Self::Project { inner, variables } => {
                f.write_str("SELECT ")?;
                for (i, var) in variables.iter().enumerate() {
                    if i > 0 {
                        f.write_str(" ")?;
                    }
                    write!(f, "{var}")?;
                }
                write!(f, " WHERE {{ {inner} }}")
            }
            Self::Distinct { inner } => {
                write!(f, "SELECT DISTINCT * WHERE {{ {inner} }}")
            }
            Self::Reduced { inner } => {
                write!(f, "SELECT REDUCED * WHERE {{ {inner} }}")
            }
            Self::Slice {
                inner,
                start,
                length,
            } => {
                write!(f, "{inner} OFFSET {start}")?;
                if let Some(length) = length {
                    write!(f, " LIMIT {length}")?;
                }
                Ok(())
            }
            Self::Group {
                inner,
                variables,
                aggregates,
            } => {
                write!(f, "{inner} GROUP BY")?;
                for var in variables {
                    write!(f, " {var}")?;
                }
                if !aggregates.is_empty() {
                    f.write_str(" HAVING")?;
                    for (var, agg) in aggregates {
                        write!(f, " ({agg} AS {var})")?;
                    }
                }
                Ok(())
            }
            Self::Service {
                name,
                inner,
                silent,
            } => {
                if *silent {
                    write!(f, "SERVICE SILENT {name} {{ {inner} }}")
                } else {
                    write!(f, "SERVICE {name} {{ {inner} }}")
                }
            }
        }
    }
}

impl GraphPattern {
    /// Formats using the SPARQL S-Expression syntax
    pub fn fmt_sse(&self, f: &mut impl fmt::Write) -> fmt::Result {
        match self {
            Self::Bgp { patterns } => {
                f.write_str("(bgp")?;
                for pattern in patterns {
                    f.write_str(" ")?;
                    pattern.fmt_sse(f)?;
                }
                f.write_str(")")
            }
            Self::Path {
                subject,
                path,
                object,
            } => {
                f.write_str("(path ")?;
                subject.fmt_sse(f)?;
                f.write_str(" ")?;
                path.fmt_sse(f)?;
                f.write_str(" ")?;
                object.fmt_sse(f)?;
                f.write_str(")")
            }
            Self::Join { left, right } => {
                f.write_str("(join ")?;
                left.fmt_sse(f)?;
                f.write_str(" ")?;
                right.fmt_sse(f)?;
                f.write_str(")")
            }
            Self::LeftJoin {
                left,
                right,
                expression,
            } => {
                f.write_str("(leftjoin ")?;
                left.fmt_sse(f)?;
                f.write_str(" ")?;
                right.fmt_sse(f)?;
                if let Some(expr) = expression {
                    f.write_str(" ")?;
                    expr.fmt_sse(f)?;
                }
                f.write_str(")")
            }
            Self::Filter { expr, inner } => {
                f.write_str("(filter ")?;
                expr.fmt_sse(f)?;
                f.write_str(" ")?;
                inner.fmt_sse(f)?;
                f.write_str(")")
            }
            Self::Union { left, right } => {
                f.write_str("(union ")?;
                left.fmt_sse(f)?;
                f.write_str(" ")?;
                right.fmt_sse(f)?;
                f.write_str(")")
            }
            Self::Graph { name, inner } => {
                f.write_str("(graph ")?;
                name.fmt_sse(f)?;
                f.write_str(" ")?;
                inner.fmt_sse(f)?;
                f.write_str(")")
            }
            Self::Extend {
                inner,
                variable,
                expression,
            } => {
                f.write_str("(extend ")?;
                inner.fmt_sse(f)?;
                f.write_str(" (")?;
                variable.fmt_sse(f)?;
                f.write_str(" ")?;
                expression.fmt_sse(f)?;
                f.write_str("))")
            }
            Self::Minus { left, right } => {
                f.write_str("(minus ")?;
                left.fmt_sse(f)?;
                f.write_str(" ")?;
                right.fmt_sse(f)?;
                f.write_str(")")
            }
            Self::Values {
                variables,
                bindings,
            } => {
                f.write_str("(table")?;
                if !variables.is_empty() {
                    f.write_str(" (vars")?;
                    for var in variables {
                        f.write_str(" ")?;
                        var.fmt_sse(f)?;
                    }
                    f.write_str(")")?;
                }
                for binding in bindings {
                    f.write_str(" (row")?;
                    for (i, value) in binding.iter().enumerate() {
                        f.write_str(" (")?;
                        variables[i].fmt_sse(f)?;
                        f.write_str(" ")?;
                        if let Some(term) = value {
                            term.fmt_sse(f)?;
                        } else {
                            f.write_str("UNDEF")?;
                        }
                        f.write_str(")")?;
                    }
                    f.write_str(")")?;
                }
                f.write_str(")")
            }
            Self::OrderBy { inner, expression } => {
                f.write_str("(order ")?;
                inner.fmt_sse(f)?;
                for expr in expression {
                    f.write_str(" ")?;
                    expr.fmt_sse(f)?;
                }
                f.write_str(")")
            }
            Self::Project { inner, variables } => {
                f.write_str("(project ")?;
                inner.fmt_sse(f)?;
                f.write_str(" (")?;
                for (i, var) in variables.iter().enumerate() {
                    if i > 0 {
                        f.write_str(" ")?;
                    }
                    var.fmt_sse(f)?;
                }
                f.write_str("))")
            }
            Self::Distinct { inner } => {
                f.write_str("(distinct ")?;
                inner.fmt_sse(f)?;
                f.write_str(")")
            }
            Self::Reduced { inner } => {
                f.write_str("(reduced ")?;
                inner.fmt_sse(f)?;
                f.write_str(")")
            }
            Self::Slice {
                inner,
                start,
                length,
            } => {
                f.write_str("(slice ")?;
                inner.fmt_sse(f)?;
                write!(f, " {start}")?;
                if let Some(length) = length {
                    write!(f, " {length}")?;
                }
                f.write_str(")")
            }
            Self::Group {
                inner,
                variables,
                aggregates,
            } => {
                f.write_str("(group ")?;
                inner.fmt_sse(f)?;
                if !variables.is_empty() {
                    f.write_str(" (")?;
                    for (i, var) in variables.iter().enumerate() {
                        if i > 0 {
                            f.write_str(" ")?;
                        }
                        var.fmt_sse(f)?;
                    }
                    f.write_str(")")?;
                }
                if !aggregates.is_empty() {
                    f.write_str(" (")?;
                    for (i, (var, agg)) in aggregates.iter().enumerate() {
                        if i > 0 {
                            f.write_str(" ")?;
                        }
                        f.write_str("(")?;
                        var.fmt_sse(f)?;
                        f.write_str(" ")?;
                        agg.fmt_sse(f)?;
                        f.write_str(")")?;
                    }
                    f.write_str(")")?;
                }
                f.write_str(")")
            }
            Self::Service {
                name,
                inner,
                silent,
            } => {
                if *silent {
                    f.write_str("(service silent ")?;
                } else {
                    f.write_str("(service ")?;
                }
                name.fmt_sse(f)?;
                f.write_str(" ")?;
                inner.fmt_sse(f)?;
                f.write_str(")")
            }
        }
    }
}

/// The union of [IRIs](https://www.w3.org/TR/rdf11-concepts/#dfn-iri), [literals](https://www.w3.org/TR/rdf11-concepts/#dfn-literal) and [triples](https://www.w3.org/TR/rdf11-concepts/#dfn-rdf-triple).
#[derive(Eq, PartialEq, Debug, Clone, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum GroundTerm {
    NamedNode(NamedNode),
    Literal(Literal),
    #[cfg(feature = "sparql-12")]
    Triple(Box<GroundTriple>),
}

impl GroundTerm {
    pub fn fmt_sse(&self, f: &mut impl fmt::Write) -> fmt::Result {
        match self {
            Self::NamedNode(node) => write!(f, "{node}"),
            Self::Literal(literal) => write!(f, "{literal}"),
            #[cfg(feature = "sparql-12")]
            Self::Triple(triple) => {
                f.write_str("<<")?;
                triple.subject.fmt_sse(f)?;
                f.write_str(" ")?;
                write!(f, "{}", triple.predicate)?;
                f.write_str(" ")?;
                triple.object.fmt_sse(f)?;
                f.write_str(">>")
            }
        }
    }
}

impl fmt::Display for GroundTerm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NamedNode(node) => node.fmt(f),
            Self::Literal(literal) => literal.fmt(f),
            #[cfg(feature = "sparql-12")]
            Self::Triple(triple) => write!(
                f,
                "<<( {} {} {} )>>",
                triple.subject, triple.predicate, triple.object
            ),
        }
    }
}

impl From<NamedNode> for GroundTerm {
    fn from(node: NamedNode) -> Self {
        Self::NamedNode(node)
    }
}

impl From<Literal> for GroundTerm {
    fn from(literal: Literal) -> Self {
        Self::Literal(literal)
    }
}

impl TryFrom<Term> for GroundTerm {
    type Error = ();

    fn try_from(term: Term) -> Result<Self, Self::Error> {
        match term {
            Term::NamedNode(t) => Ok(t.into()),
            Term::BlankNode(_) => Err(()), // Blank nodes not allowed in ground terms
            Term::Literal(t) => Ok(t.into()),
            Term::Variable(_) => Err(()), // Variables not allowed in ground terms
            Term::QuotedTriple(_) => Err(()), // Quoted triples not yet supported
        }
    }
}

impl From<GroundTerm> for Term {
    fn from(term: GroundTerm) -> Self {
        match term {
            GroundTerm::NamedNode(t) => t.into(),
            GroundTerm::Literal(l) => l.into(),
            #[cfg(feature = "sparql-12")]
            GroundTerm::Triple(t) => {
                // Convert GroundTriple to QuotedTriple
                // For now, we create a basic triple representation
                // Full RDF-star support would require proper conversion
                use crate::model::star::QuotedTriple;
                use crate::model::Triple as ModelTriple;

                // Convert GroundSubject to Subject
                let subject: crate::model::Subject = match t.subject {
                    GroundSubject::NamedNode(n) => n.into(),
                    GroundSubject::Triple(_) => {
                        // Nested triples - not fully supported yet
                        // Use a placeholder for now
                        return Term::NamedNode(crate::model::NamedNode::new_unchecked("http://example.org/unsupported-nested-triple"));
                    }
                };

                let predicate: crate::model::Predicate = t.predicate.into();

                // Convert GroundTerm to Object
                let object: crate::model::Object = match t.object {
                    GroundTerm::NamedNode(n) => n.into(),
                    GroundTerm::Literal(l) => l.into(),
                    GroundTerm::Triple(_) => {
                        // Nested triples - not fully supported yet
                        return Term::NamedNode(crate::model::NamedNode::new_unchecked("http://example.org/unsupported-nested-triple"));
                    }
                };

                let triple = ModelTriple::new(subject, predicate, object);
                Term::QuotedTriple(Box::new(QuotedTriple::new(triple)))
            }
        }
    }
}

/// A [RDF triple](https://www.w3.org/TR/rdf11-concepts/#dfn-rdf-triple) without blank nodes.
#[cfg(feature = "sparql-12")]
#[derive(Eq, PartialEq, Debug, Clone, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GroundTriple {
    pub subject: GroundSubject,
    pub predicate: NamedNode,
    pub object: GroundTerm,
}

#[cfg(feature = "sparql-12")]
impl GroundTriple {
    pub fn fmt_sse(&self, f: &mut impl fmt::Write) -> fmt::Result {
        f.write_str("(triple ")?;
        self.subject.fmt_sse(f)?;
        f.write_str(" ")?;
        write!(f, "{}", self.predicate)?;
        f.write_str(" ")?;
        self.object.fmt_sse(f)?;
        f.write_str(")")
    }
}

/// Either a named node or a quoted triple
#[cfg(feature = "sparql-12")]
#[derive(Eq, PartialEq, Debug, Clone, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum GroundSubject {
    NamedNode(NamedNode),
    Triple(Box<GroundTriple>),
}

#[cfg(feature = "sparql-12")]
impl GroundSubject {
    pub fn fmt_sse(&self, f: &mut impl fmt::Write) -> fmt::Result {
        match self {
            Self::NamedNode(node) => write!(f, "{node}"),
            Self::Triple(triple) => {
                f.write_str("<<")?;
                triple.fmt_sse(f)?;
                f.write_str(">>")
            }
        }
    }
}

#[cfg(feature = "sparql-12")]
impl fmt::Display for GroundSubject {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NamedNode(node) => write!(f, "{}", node),
            Self::Triple(triple) => write!(f, "<<( {} {} {} )>>", triple.subject, triple.predicate, triple.object),
        }
    }
}

/// A triple pattern
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TriplePattern {
    pub subject: TermPattern,
    pub predicate: TermPattern,
    pub object: TermPattern,
}

impl TriplePattern {
    pub fn new(
        subject: impl Into<TermPattern>,
        predicate: impl Into<TermPattern>,
        object: impl Into<TermPattern>,
    ) -> Self {
        Self {
            subject: subject.into(),
            predicate: predicate.into(),
            object: object.into(),
        }
    }

    pub fn fmt_sse(&self, f: &mut impl fmt::Write) -> fmt::Result {
        f.write_str("(triple ")?;
        self.subject.fmt_sse(f)?;
        f.write_str(" ")?;
        self.predicate.fmt_sse(f)?;
        f.write_str(" ")?;
        self.object.fmt_sse(f)?;
        f.write_str(")")
    }
}

impl fmt::Display for TriplePattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {} {}", self.subject, self.predicate, self.object)
    }
}

/// A term pattern that can be either a concrete term or a variable
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TermPattern {
    NamedNode(NamedNode),
    BlankNode(BlankNode),
    Literal(Literal),
    Variable(Variable),
    #[cfg(feature = "sparql-12")]
    Triple(Box<TriplePattern>),
}

impl TermPattern {
    pub fn fmt_sse(&self, f: &mut impl fmt::Write) -> fmt::Result {
        match self {
            Self::NamedNode(node) => write!(f, "{node}"),
            Self::BlankNode(node) => write!(f, "{node}"),
            Self::Literal(literal) => write!(f, "{literal}"),
            Self::Variable(var) => write!(f, "{var}"),
            #[cfg(feature = "sparql-12")]
            Self::Triple(triple) => {
                f.write_str("<<")?;
                triple.fmt_sse(f)?;
                f.write_str(">>")
            }
        }
    }
}

impl fmt::Display for TermPattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NamedNode(node) => node.fmt(f),
            Self::BlankNode(node) => node.fmt(f),
            Self::Literal(literal) => literal.fmt(f),
            Self::Variable(var) => var.fmt(f),
            #[cfg(feature = "sparql-12")]
            Self::Triple(triple) => write!(f, "<<{triple}>>"),
        }
    }
}

impl From<Variable> for TermPattern {
    fn from(v: Variable) -> Self {
        TermPattern::Variable(v)
    }
}

impl From<NamedNode> for TermPattern {
    fn from(n: NamedNode) -> Self {
        TermPattern::NamedNode(n)
    }
}

impl From<BlankNode> for TermPattern {
    fn from(n: BlankNode) -> Self {
        TermPattern::BlankNode(n)
    }
}

impl From<Literal> for TermPattern {
    fn from(l: Literal) -> Self {
        TermPattern::Literal(l)
    }
}

/// A named node pattern (can be a concrete named node or a variable)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum NamedNodePattern {
    NamedNode(NamedNode),
    Variable(Variable),
}

impl NamedNodePattern {
    pub fn fmt_sse(&self, f: &mut impl fmt::Write) -> fmt::Result {
        match self {
            Self::NamedNode(node) => write!(f, "{node}"),
            Self::Variable(var) => write!(f, "{var}"),
        }
    }
}

impl fmt::Display for NamedNodePattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NamedNode(node) => node.fmt(f),
            Self::Variable(var) => var.fmt(f),
        }
    }
}

impl From<NamedNode> for NamedNodePattern {
    fn from(node: NamedNode) -> Self {
        Self::NamedNode(node)
    }
}

impl From<Variable> for NamedNodePattern {
    fn from(var: Variable) -> Self {
        Self::Variable(var)
    }
}

/// An order expression
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum OrderExpression {
    /// Ascending order
    Asc(Expression),
    /// Descending order
    Desc(Expression),
}

impl OrderExpression {
    pub fn fmt_sse(&self, f: &mut impl fmt::Write) -> fmt::Result {
        match self {
            Self::Asc(expr) => {
                f.write_str("(asc ")?;
                expr.fmt_sse(f)?;
                f.write_str(")")
            }
            Self::Desc(expr) => {
                f.write_str("(desc ")?;
                expr.fmt_sse(f)?;
                f.write_str(")")
            }
        }
    }
}

impl fmt::Display for OrderExpression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Asc(expr) => write!(f, "ASC({expr})"),
            Self::Desc(expr) => write!(f, "DESC({expr})"),
        }
    }
}

/// An aggregate expression
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum AggregateExpression {
    Count {
        expr: Option<Box<Expression>>,
        distinct: bool,
    },
    Sum {
        expr: Box<Expression>,
        distinct: bool,
    },
    Avg {
        expr: Box<Expression>,
        distinct: bool,
    },
    Min {
        expr: Box<Expression>,
        distinct: bool,
    },
    Max {
        expr: Box<Expression>,
        distinct: bool,
    },
    GroupConcat {
        expr: Box<Expression>,
        distinct: bool,
        separator: Option<String>,
    },
    Sample {
        expr: Box<Expression>,
        distinct: bool,
    },
    Custom {
        name: NamedNode,
        expr: Box<Expression>,
        distinct: bool,
    },
}

impl AggregateExpression {
    pub fn fmt_sse(&self, f: &mut impl fmt::Write) -> fmt::Result {
        match self {
            Self::Count { expr, distinct } => {
                if *distinct {
                    f.write_str("(count distinct")?;
                } else {
                    f.write_str("(count")?;
                }
                if let Some(expr) = expr {
                    f.write_str(" ")?;
                    expr.fmt_sse(f)?;
                }
                f.write_str(")")
            }
            Self::Sum { expr, distinct } => {
                if *distinct {
                    f.write_str("(sum distinct ")?;
                } else {
                    f.write_str("(sum ")?;
                }
                expr.fmt_sse(f)?;
                f.write_str(")")
            }
            Self::Avg { expr, distinct } => {
                if *distinct {
                    f.write_str("(avg distinct ")?;
                } else {
                    f.write_str("(avg ")?;
                }
                expr.fmt_sse(f)?;
                f.write_str(")")
            }
            Self::Min { expr, distinct } => {
                if *distinct {
                    f.write_str("(min distinct ")?;
                } else {
                    f.write_str("(min ")?;
                }
                expr.fmt_sse(f)?;
                f.write_str(")")
            }
            Self::Max { expr, distinct } => {
                if *distinct {
                    f.write_str("(max distinct ")?;
                } else {
                    f.write_str("(max ")?;
                }
                expr.fmt_sse(f)?;
                f.write_str(")")
            }
            Self::GroupConcat {
                expr,
                distinct,
                separator,
            } => {
                if *distinct {
                    f.write_str("(group_concat distinct ")?;
                } else {
                    f.write_str("(group_concat ")?;
                }
                expr.fmt_sse(f)?;
                if let Some(sep) = separator {
                    write!(f, " \"{sep}\"")?;
                }
                f.write_str(")")
            }
            Self::Sample { expr, distinct } => {
                if *distinct {
                    f.write_str("(sample distinct ")?;
                } else {
                    f.write_str("(sample ")?;
                }
                expr.fmt_sse(f)?;
                f.write_str(")")
            }
            Self::Custom {
                name,
                expr,
                distinct,
            } => {
                if *distinct {
                    write!(f, "({name} distinct ")?;
                } else {
                    write!(f, "({name} ")?;
                }
                expr.fmt_sse(f)?;
                f.write_str(")")
            }
        }
    }
}

impl fmt::Display for AggregateExpression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Count { expr, distinct } => {
                if *distinct {
                    f.write_str("COUNT(DISTINCT ")?;
                } else {
                    f.write_str("COUNT(")?;
                }
                if let Some(expr) = expr {
                    expr.fmt(f)?;
                } else {
                    f.write_str("*")?;
                }
                f.write_str(")")
            }
            Self::Sum { expr, distinct } => {
                if *distinct {
                    f.write_str("SUM(DISTINCT ")?;
                } else {
                    f.write_str("SUM(")?;
                }
                expr.fmt(f)?;
                f.write_str(")")
            }
            Self::Avg { expr, distinct } => {
                if *distinct {
                    f.write_str("AVG(DISTINCT ")?;
                } else {
                    f.write_str("AVG(")?;
                }
                expr.fmt(f)?;
                f.write_str(")")
            }
            Self::Min { expr, distinct } => {
                if *distinct {
                    f.write_str("MIN(DISTINCT ")?;
                } else {
                    f.write_str("MIN(")?;
                }
                expr.fmt(f)?;
                f.write_str(")")
            }
            Self::Max { expr, distinct } => {
                if *distinct {
                    f.write_str("MAX(DISTINCT ")?;
                } else {
                    f.write_str("MAX(")?;
                }
                expr.fmt(f)?;
                f.write_str(")")
            }
            Self::GroupConcat {
                expr,
                distinct,
                separator,
            } => {
                if *distinct {
                    f.write_str("GROUP_CONCAT(DISTINCT ")?;
                } else {
                    f.write_str("GROUP_CONCAT(")?;
                }
                expr.fmt(f)?;
                if let Some(sep) = separator {
                    write!(f, "; SEPARATOR=\"{sep}\"")?;
                }
                f.write_str(")")
            }
            Self::Sample { expr, distinct } => {
                if *distinct {
                    f.write_str("SAMPLE(DISTINCT ")?;
                } else {
                    f.write_str("SAMPLE(")?;
                }
                expr.fmt(f)?;
                f.write_str(")")
            }
            Self::Custom {
                name,
                expr,
                distinct,
            } => {
                if *distinct {
                    write!(f, "{name}(DISTINCT ")?;
                } else {
                    write!(f, "{name}(")?;
                }
                expr.fmt(f)?;
                f.write_str(")")
            }
        }
    }
}

// Helper functions for S-Expression formatting
fn fmt_sse_binary_expression(
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

fn fmt_sse_unary_expression(
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{NamedNode, Variable};

    #[test]
    fn test_property_path_display() {
        let p1 = NamedNode::new("http://example.org/p1").unwrap();
        let p2 = NamedNode::new("http://example.org/p2").unwrap();

        let path = PropertyPathExpression::Sequence(
            Box::new(PropertyPathExpression::NamedNode(p1)),
            Box::new(PropertyPathExpression::NamedNode(p2)),
        );

        assert!(path.to_string().contains("/"));
    }

    #[test]
    fn test_basic_graph_pattern() {
        let subject = TermPattern::Variable(Variable::new("s").unwrap());
        let predicate = TermPattern::Variable(Variable::new("p").unwrap());
        let object = TermPattern::Variable(Variable::new("o").unwrap());

        let triple = TriplePattern::new(subject, predicate, object);
        let bgp = GraphPattern::Bgp {
            patterns: vec![triple],
        };

        let mut sse = String::new();
        bgp.fmt_sse(&mut sse).unwrap();
        assert!(sse.contains("bgp"));
        assert!(sse.contains("?s"));
        assert!(sse.contains("?p"));
        assert!(sse.contains("?o"));
    }

    #[test]
    fn test_expression_formatting() {
        let var1 = Expression::Variable(Variable::new("x").unwrap());
        let var2 = Expression::Variable(Variable::new("y").unwrap());
        let expr = Expression::Add(Box::new(var1), Box::new(var2));

        let mut sse = String::new();
        expr.fmt_sse(&mut sse).unwrap();
        assert!(sse.contains("+ ?x ?y"));
    }

    #[test]
    fn test_built_in_function() {
        let func = BuiltInFunction::Str;
        assert_eq!(func.to_string(), "STR");

        let mut sse = String::new();
        func.fmt_sse(&mut sse).unwrap();
        assert_eq!(sse, "str");
    }
}
