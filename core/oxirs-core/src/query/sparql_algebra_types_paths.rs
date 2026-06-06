//! Property path expression algebra node.
//!
//! Contains [`PropertyPathExpression`] and its `Display`/SSE implementations.

use crate::model::*;
use std::fmt;

// ────────────────────────────────────────────────────────────────────────────
// PropertyPathExpression
// ────────────────────────────────────────────────────────────────────────────

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
