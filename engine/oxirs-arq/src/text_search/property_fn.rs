//! `text:query` SPARQL property function backed by the tantivy text index.
//!
//! ## Usage in SPARQL
//!
//! ```sparql
//! PREFIX text: <http://jena.apache.org/text#>
//!
//! SELECT ?doc WHERE {
//!   ?doc text:query ("semantic web") .
//!   ?doc rdf:type ex:Document .
//! }
//! ```
//!
//! Argument forms (Jena-compatible):
//! - `(queryString)` — search all indexed literals
//! - `(queryString maxResults)` — search with explicit limit
//! - `(predicateIri queryString)` — predicate-filtered search
//! - `(predicateIri queryString maxResults)` — predicate-filtered with limit

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{anyhow, bail, Result};

use crate::algebra::{Iri, Term as AlgebraTerm, Variable};
use crate::property_functions::{
    PropFuncArg, PropertyFunction, PropertyFunctionContext, PropertyFunctionResult,
};

use super::index::TextSearchIndex;

/// IRI of the `text:query` property function (Jena namespace)
pub const TEXT_QUERY_IRI: &str = "http://jena.apache.org/text#query";

// ---------------------------------------------------------------------------
// TextQueryPropertyFunction
// ---------------------------------------------------------------------------

/// SPARQL property function implementing `text:query`.
///
/// Dispatches full-text searches against a shared [`TextSearchIndex`] and
/// returns SPARQL solution mappings binding the subject variable to each
/// matching subject IRI (ranked by BM25 score).
pub struct TextQueryPropertyFunction {
    index: Arc<TextSearchIndex>,
}

impl TextQueryPropertyFunction {
    /// Wrap an existing shared index.
    pub fn new(index: Arc<TextSearchIndex>) -> Self {
        Self { index }
    }

    /// The Jena text namespace.
    pub fn text_namespace() -> &'static str {
        super::index::TEXT_NAMESPACE
    }

    /// The full IRI of the `text:query` property function.
    pub fn iri() -> &'static str {
        TEXT_QUERY_IRI
    }
}

impl PropertyFunction for TextQueryPropertyFunction {
    fn uri(&self) -> &str {
        TEXT_QUERY_IRI
    }

    fn build(
        &self,
        _subject: &PropFuncArg,
        predicate: &str,
        object: &PropFuncArg,
        _context: &PropertyFunctionContext,
    ) -> Result<()> {
        if predicate != self.uri() {
            bail!(
                "Predicate mismatch: expected {}, got {}",
                self.uri(),
                predicate
            );
        }
        // Object must be a non-empty list or a single literal node
        match object {
            PropFuncArg::List(items) if items.is_empty() => {
                bail!("text:query requires at least one argument (query string)")
            }
            PropFuncArg::List(_) | PropFuncArg::Node(_) => Ok(()),
        }
    }

    /// Execute the text:query property function.
    ///
    /// Argument forms (object position):
    /// - `List([Literal(query)])` — full-text search, default max_results = 10
    /// - `List([Literal(query), Literal(maxResults)])` — with explicit limit
    /// - `List([Iri(predicate), Literal(query)])` — predicate-filtered
    /// - `List([Iri(predicate), Literal(query), Literal(maxResults)])` — both
    fn execute(
        &self,
        subject: &PropFuncArg,
        _predicate: &str,
        object: &PropFuncArg,
        context: &PropertyFunctionContext,
    ) -> Result<PropertyFunctionResult> {
        let subject = context.substitute(subject);
        let object = context.substitute(object);

        // Determine what variable (or concrete IRI) is in the subject position
        let subject_var: Option<Variable> = match &subject {
            PropFuncArg::Node(AlgebraTerm::Variable(v)) => Some(v.clone()),
            PropFuncArg::Node(AlgebraTerm::Iri(_)) => None, // bound subject — filter mode
            _ => bail!("text:query: subject must be a variable or an IRI"),
        };
        let bound_subject: Option<String> = match &subject {
            PropFuncArg::Node(AlgebraTerm::Iri(iri)) => Some(iri.as_str().to_string()),
            _ => None,
        };

        // Parse the argument list
        let args: Vec<AlgebraTerm> = match &object {
            PropFuncArg::List(items) => items.clone(),
            PropFuncArg::Node(AlgebraTerm::Literal(lit)) => {
                // Treat single literal node as a 1-element list
                vec![AlgebraTerm::Literal(lit.clone())]
            }
            _ => bail!("text:query: object must be a list of arguments"),
        };

        // Dispatch based on argument count and types
        match args.as_slice() {
            // (queryString)
            [AlgebraTerm::Literal(lit)] => {
                self.execute_search(&subject_var, bound_subject.as_deref(), &lit.value, None, 10)
            }
            // (queryString, maxResults)
            [AlgebraTerm::Literal(query_lit), AlgebraTerm::Literal(max_lit)] => {
                let max = parse_max_results(&max_lit.value)?;
                self.execute_search(
                    &subject_var,
                    bound_subject.as_deref(),
                    &query_lit.value,
                    None,
                    max,
                )
            }
            // (predicateIri, queryString)
            [AlgebraTerm::Iri(pred_iri), AlgebraTerm::Literal(query_lit)] => self.execute_search(
                &subject_var,
                bound_subject.as_deref(),
                &query_lit.value,
                Some(pred_iri.as_str()),
                10,
            ),
            // (predicateIri, queryString, maxResults)
            [AlgebraTerm::Iri(pred_iri), AlgebraTerm::Literal(query_lit), AlgebraTerm::Literal(max_lit)] =>
            {
                let max = parse_max_results(&max_lit.value)?;
                self.execute_search(
                    &subject_var,
                    bound_subject.as_deref(),
                    &query_lit.value,
                    Some(pred_iri.as_str()),
                    max,
                )
            }
            _ => bail!(
                "text:query: unrecognised argument pattern — expected 1–3 args: \
                 [predIri?] queryString [maxResults?]"
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

impl TextQueryPropertyFunction {
    fn execute_search(
        &self,
        subject_var: &Option<Variable>,
        bound_subject: Option<&str>,
        query_str: &str,
        predicate_filter: Option<&str>,
        max_results: usize,
    ) -> Result<PropertyFunctionResult> {
        let hits = if let Some(pred) = predicate_filter {
            self.index
                .search_predicate(query_str, pred, max_results)
                .map_err(|e| anyhow!("text:query search error: {e}"))?
        } else {
            self.index
                .search(query_str, max_results)
                .map_err(|e| anyhow!("text:query search error: {e}"))?
        };

        match (subject_var, bound_subject) {
            // Subject is a variable — generate one binding per hit
            (Some(var), None) => {
                let solutions: Vec<HashMap<Variable, AlgebraTerm>> = hits
                    .into_iter()
                    .filter_map(|hit| {
                        Iri::new(&hit.subject_iri).ok().map(|iri| {
                            let mut bindings = HashMap::new();
                            bindings.insert(var.clone(), AlgebraTerm::Iri(iri));
                            bindings
                        })
                    })
                    .collect();
                Ok(PropertyFunctionResult::Multiple(solutions))
            }
            // Subject is bound — test whether it appears in the results
            (None, Some(bound_iri)) => {
                let matched = hits.iter().any(|hit| hit.subject_iri == bound_iri);
                Ok(PropertyFunctionResult::Boolean(matched))
            }
            _ => bail!("text:query: internal argument state error"),
        }
    }
}

/// Parse a max-results integer from a SPARQL literal string.
fn parse_max_results(s: &str) -> Result<usize> {
    s.trim().parse::<usize>().map_err(|_| {
        anyhow!(
            "text:query: maxResults must be a non-negative integer, got {:?}",
            s
        )
    })
}
