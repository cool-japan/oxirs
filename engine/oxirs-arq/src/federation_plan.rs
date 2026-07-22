//! Federation execution plan tree and SPARQL-serialization helpers.
//!
//! Split out of `federation.rs` to keep that file under the 2000-line limit.
//! These items support [`crate::federation::FederationExecutor`]: the plan tree
//! preserves the algebra combinator joining SERVICE clauses (so conjunctive
//! multi-SERVICE queries are relationally joined, not UNION-merged), and the
//! serialization helpers render terms/expressions/paths into real SPARQL for
//! pushdown to remote endpoints.

use crate::algebra::{Binding, Solution};
use crate::federation::FederatedSubquery;
use anyhow::Result;

/// A federation execution plan preserving the algebra combinator that connected
/// the SERVICE clauses, so a conjunctive multi-SERVICE query is relationally
/// joined on shared variables rather than UNION-merged.
#[derive(Debug, Clone)]
pub enum FederationPlan {
    /// A single SERVICE subquery executed against one endpoint.
    Service(Box<FederatedSubquery>),
    /// Inner join of two sub-plans on their shared variables.
    Join(Box<FederationPlan>, Box<FederationPlan>),
    /// Union (disjunction) of two sub-plans.
    Union(Box<FederationPlan>, Box<FederationPlan>),
    /// Left (OPTIONAL) join of two sub-plans.
    LeftJoin(Box<FederationPlan>, Box<FederationPlan>),
    /// MINUS of two sub-plans.
    Minus(Box<FederationPlan>, Box<FederationPlan>),
}

/// The combinator used when merging two branch plans.
#[derive(Debug, Clone, Copy)]
pub(crate) enum CombineOp {
    Join,
    Union,
    LeftJoin,
    Minus,
}

impl CombineOp {
    pub(crate) fn build(self, left: FederationPlan, right: FederationPlan) -> FederationPlan {
        let (l, r) = (Box::new(left), Box::new(right));
        match self {
            CombineOp::Join => FederationPlan::Join(l, r),
            CombineOp::Union => FederationPlan::Union(l, r),
            CombineOp::LeftJoin => FederationPlan::LeftJoin(l, r),
            CombineOp::Minus => FederationPlan::Minus(l, r),
        }
    }
}

/// Merge two bindings if they are join-compatible (agree on every shared
/// variable), returning the merged binding, or `None` if incompatible.
pub(crate) fn merge_compatible_bindings(left: &Binding, right: &Binding) -> Option<Binding> {
    let mut merged = left.clone();
    for (var, value) in right {
        match merged.get(var) {
            Some(existing) if existing != value => return None,
            Some(_) => {}
            None => {
                merged.insert(var.clone(), value.clone());
            }
        }
    }
    Some(merged)
}

/// Inner join of two solutions on their shared variables (nested-loop).
pub(crate) fn join_solutions(left: &Solution, right: &Solution) -> Solution {
    let mut out = Solution::new();
    for l in left {
        for r in right {
            if let Some(merged) = merge_compatible_bindings(l, r) {
                out.push(merged);
            }
        }
    }
    out
}

/// Left (OPTIONAL) join: every left row is kept; matching right rows extend it.
pub(crate) fn left_join_solutions(left: &Solution, right: &Solution) -> Solution {
    let mut out = Solution::new();
    for l in left {
        let mut matched = false;
        for r in right {
            if let Some(merged) = merge_compatible_bindings(l, r) {
                out.push(merged);
                matched = true;
            }
        }
        if !matched {
            out.push(l.clone());
        }
    }
    out
}

/// SPARQL MINUS: drop a left row only when a right row shares ≥1 variable with
/// it AND agrees on every shared variable (disjoint domains remove nothing).
pub(crate) fn minus_solutions(left: &Solution, right: &Solution) -> Solution {
    left.iter()
        .filter(|l| !right.iter().any(|r| minus_removes(l, r)))
        .cloned()
        .collect()
}

pub(crate) fn minus_removes(left: &Binding, right: &Binding) -> bool {
    let mut shares_variable = false;
    for (var, left_value) in left {
        if let Some(right_value) = right.get(var) {
            if left_value != right_value {
                return false;
            }
            shares_variable = true;
        }
    }
    shares_variable
}

/// Escape a lexical form for inclusion in a SPARQL double-quoted string literal
/// (SPARQL 1.1 grammar, STRING_LITERAL_QUOTE + ECHAR).
pub(crate) fn escape_sparql_string(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            _ => out.push(ch),
        }
    }
    out
}

/// Serialize an algebra literal to SPARQL, preserving datatype (`^^<iri>`) or
/// language tag (`@lang`) and escaping the lexical form.
pub(crate) fn literal_to_sparql(lit: &crate::algebra::Literal) -> String {
    let escaped = escape_sparql_string(&lit.value);
    if let Some(lang) = &lit.language {
        format!("\"{escaped}\"@{lang}")
    } else if let Some(dt) = &lit.datatype {
        let dt_str = dt.as_str();
        // xsd:string is the implicit datatype of a plain literal; omit it.
        if dt_str == "http://www.w3.org/2001/XMLSchema#string" {
            format!("\"{escaped}\"")
        } else {
            format!("\"{escaped}\"^^<{dt_str}>")
        }
    } else {
        format!("\"{escaped}\"")
    }
}

/// SPARQL operator symbol for a binary operator, or `None` for operators that
/// use function-call / keyword syntax (`sameTerm`, `IN`, `NOT IN`).
pub(crate) fn binary_operator_symbol(op: &crate::algebra::BinaryOperator) -> Option<&'static str> {
    use crate::algebra::BinaryOperator as B;
    Some(match op {
        B::Add => "+",
        B::Subtract => "-",
        B::Multiply => "*",
        B::Divide => "/",
        B::Equal => "=",
        B::NotEqual => "!=",
        B::Less => "<",
        B::LessEqual => "<=",
        B::Greater => ">",
        B::GreaterEqual => ">=",
        B::And => "&&",
        B::Or => "||",
        B::SameTerm | B::In | B::NotIn => return None,
    })
}

/// Serialize a property path to SPARQL 1.1 property-path syntax.
pub(crate) fn property_path_to_sparql(path: &crate::algebra::PropertyPath) -> Result<String> {
    use crate::algebra::PropertyPath as P;
    Ok(match path {
        P::Iri(iri) => format!("<{}>", iri.as_str()),
        P::Variable(v) => format!("?{}", v.name()),
        P::Inverse(inner) => format!("^{}", property_path_to_sparql(inner)?),
        P::Sequence(a, b) => format!(
            "({}/{})",
            property_path_to_sparql(a)?,
            property_path_to_sparql(b)?
        ),
        P::Alternative(a, b) => format!(
            "({}|{})",
            property_path_to_sparql(a)?,
            property_path_to_sparql(b)?
        ),
        P::ZeroOrMore(inner) => format!("({})*", property_path_to_sparql(inner)?),
        P::OneOrMore(inner) => format!("({})+", property_path_to_sparql(inner)?),
        P::ZeroOrOne(inner) => format!("({})?", property_path_to_sparql(inner)?),
        P::NegatedPropertySet(paths) => {
            let inner: Result<Vec<String>> = paths.iter().map(property_path_to_sparql).collect();
            format!("!({})", inner?.join("|"))
        }
    })
}
