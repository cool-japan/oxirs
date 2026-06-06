//! Triple Pattern Fragments query parsing.
//!
//! A TPF query is a triple pattern composed of subject, predicate and object
//! components, each of which may be bound (concrete IRI/blank node/literal)
//! or unbound (variable). This module provides minimal validation suited to
//! the TPF specification.

use super::TpfQueryParams;

/// A parsed Triple Pattern Fragment query.
///
/// Each component is `None` when unbound (a variable in the pattern). When
/// bound, the value is the raw string supplied by the client; IRIs are
/// validated by [`parse_tpf_query`] but no IRI normalization is performed.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TpfQuery {
    /// Subject component (IRI or blank node identifier such as `_:b0`).
    pub subject: Option<String>,
    /// Predicate component (IRI only — RDF forbids blank node predicates).
    pub predicate: Option<String>,
    /// Object component (IRI, blank node identifier, or RDF literal).
    pub object: Option<String>,
}

impl TpfQuery {
    /// Returns `true` when none of the components are bound.
    ///
    /// An unbound query matches every triple in the underlying store, paged
    /// according to the request's pagination parameters.
    pub fn is_unbound(&self) -> bool {
        self.subject.is_none() && self.predicate.is_none() && self.object.is_none()
    }

    /// Returns the number of bound components in the pattern (0 to 3).
    pub fn count_bound(&self) -> usize {
        self.subject.iter().count() + self.predicate.iter().count() + self.object.iter().count()
    }
}

/// Parse query parameters into a [`TpfQuery`].
///
/// IRIs are validated minimally: subject and predicate must begin with
/// `http://`, `https://`, and (for subject) blank node identifiers prefixed
/// with `_:` are also accepted. Empty strings are normalised to unbound
/// components.
pub fn parse_tpf_query(params: &TpfQueryParams) -> Result<TpfQuery, String> {
    let q = TpfQuery {
        subject: params.subject.clone().filter(|s| !s.is_empty()),
        predicate: params.predicate.clone().filter(|s| !s.is_empty()),
        object: params.object.clone().filter(|s| !s.is_empty()),
    };

    if let Some(s) = &q.subject {
        if !s.starts_with("http://") && !s.starts_with("https://") && !s.starts_with("_:") {
            return Err(format!("invalid subject IRI: {}", s));
        }
    }
    if let Some(p) = &q.predicate {
        if !p.starts_with("http://") && !p.starts_with("https://") {
            return Err(format!("invalid predicate IRI: {}", p));
        }
    }

    Ok(q)
}
