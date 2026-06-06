//! Term, ground-term, ordering and aggregate algebra nodes.
//!
//! Contains [`GroundTerm`], [`GroundTriple`], [`GroundSubject`], [`TriplePattern`],
//! [`TermPattern`], [`NamedNodePattern`], [`OrderExpression`] and
//! [`AggregateExpression`] together with their `Display`/SSE implementations.

use super::sparql_algebra_types_expr::Expression;
use crate::model::*;
use std::fmt;

// ────────────────────────────────────────────────────────────────────────────
// GroundTerm / GroundTriple / GroundSubject
// ────────────────────────────────────────────────────────────────────────────

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
                        return Term::NamedNode(crate::model::NamedNode::new_unchecked(
                            "http://example.org/unsupported-nested-triple",
                        ));
                    }
                };

                let predicate: crate::model::Predicate = t.predicate.into();

                // Convert GroundTerm to Object
                let object: crate::model::Object = match t.object {
                    GroundTerm::NamedNode(n) => n.into(),
                    GroundTerm::Literal(l) => l.into(),
                    GroundTerm::Triple(_) => {
                        // Nested triples - not fully supported yet
                        return Term::NamedNode(crate::model::NamedNode::new_unchecked(
                            "http://example.org/unsupported-nested-triple",
                        ));
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
            Self::Triple(triple) => write!(
                f,
                "<<( {} {} {} )>>",
                triple.subject, triple.predicate, triple.object
            ),
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// TriplePattern / TermPattern / NamedNodePattern
// ────────────────────────────────────────────────────────────────────────────

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

// ────────────────────────────────────────────────────────────────────────────
// OrderExpression
// ────────────────────────────────────────────────────────────────────────────

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

// ────────────────────────────────────────────────────────────────────────────
// AggregateExpression
// ────────────────────────────────────────────────────────────────────────────

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
