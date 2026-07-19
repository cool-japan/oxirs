//! SPARQL Query Router for hybrid RDF + time-series queries
//!
//! This module routes SPARQL queries to the appropriate backend
//! and merges results from RDF and time-series storage.

use crate::error::TsdbResult;
use oxirs_core::model::{Literal, Term};
use oxirs_core::rdf_store::{OxirsQueryResults, QueryResults, VariableBinding};
use std::collections::HashMap;

/// Query routing decision
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RoutingDecision {
    /// Route to RDF store only
    RdfOnly,
    /// Route to time-series engine only
    TimeseriesOnly,
    /// Route to both and merge results
    Hybrid,
}

/// SPARQL query router
#[derive(Debug)]
pub struct QueryRouter {
    /// Temporal function prefixes
    temporal_prefixes: Vec<String>,
}

impl QueryRouter {
    /// Create a new query router
    pub fn new() -> Self {
        Self {
            temporal_prefixes: vec![
                "ts:window".to_string(),
                "ts:resample".to_string(),
                "ts:interpolate".to_string(),
            ],
        }
    }

    /// Analyze a SPARQL query and determine routing
    ///
    /// Detection heuristics:
    /// 1. Contains temporal functions (ts:window, ts:resample, etc.) → Hybrid
    /// 2. Only queries metadata predicates (rdf:type, rdfs:label) → RdfOnly
    /// 3. Only queries time-series predicates (qudt:numericValue) → TimeseriesOnly
    /// 4. Mixed query → Hybrid
    pub fn route(&self, sparql: &str) -> TsdbResult<RoutingDecision> {
        let sparql_lower = sparql.to_lowercase();

        // Check for temporal functions
        let has_temporal_functions = self
            .temporal_prefixes
            .iter()
            .any(|prefix| sparql_lower.contains(&prefix.to_lowercase()));

        if has_temporal_functions {
            return Ok(RoutingDecision::Hybrid);
        }

        // Check for time-series predicates
        let has_ts_predicates = sparql_lower.contains("numericvalue")
            || sparql_lower.contains("hassimpleresult")
            || sparql_lower.contains("ts:value");

        // Check for metadata predicates
        let has_metadata_predicates = sparql_lower.contains("rdf:type")
            || sparql_lower.contains("rdfs:label")
            || sparql_lower.contains("rdfs:comment");

        match (has_ts_predicates, has_metadata_predicates) {
            (true, false) => Ok(RoutingDecision::TimeseriesOnly),
            (false, true) => Ok(RoutingDecision::RdfOnly),
            (true, true) => Ok(RoutingDecision::Hybrid),
            (false, false) => Ok(RoutingDecision::RdfOnly), // Default to RDF
        }
    }

    /// Merge results from RDF and time-series queries
    ///
    /// Each entry of `ts_results` is one time-series solution mapping
    /// (variable name -> lexical value, e.g. from a `ts:window`/`ts:resample`
    /// projection). These are joined against `rdf_results`'s bindings on
    /// shared variables using standard SPARQL join semantics: a pair of
    /// rows is kept only if every variable they have in common agrees,
    /// and the combined row carries the union of both sides' bindings.
    ///
    /// If `rdf_results` is not a SELECT (`Bindings`) result -- e.g. ASK or
    /// CONSTRUCT -- there is no tabular shape to join against, so the
    /// time-series rows are returned directly as the query's bindings
    /// (the common case for a `TimeseriesOnly`-shaped Hybrid query).
    pub fn merge_results(
        &self,
        rdf_results: OxirsQueryResults,
        ts_results: Vec<HashMap<String, String>>,
    ) -> TsdbResult<OxirsQueryResults> {
        if ts_results.is_empty() {
            return Ok(rdf_results);
        }

        let ts_bindings: Vec<VariableBinding> = ts_results
            .into_iter()
            .map(|row| {
                let mut binding = VariableBinding::new();
                for (var, value) in row {
                    binding.bind(var, Term::Literal(Literal::new(value)));
                }
                binding
            })
            .collect();

        let QueryResults::Bindings(rdf_bindings) = rdf_results.results() else {
            // Non-tabular RDF result shape (ASK/CONSTRUCT): surface the
            // time-series rows directly rather than silently discarding them.
            let mut variables: Vec<String> = Vec::new();
            for binding in &ts_bindings {
                for var in binding.bindings.keys() {
                    if !variables.contains(var) {
                        variables.push(var.clone());
                    }
                }
            }
            return Ok(OxirsQueryResults::from_bindings(ts_bindings, variables));
        };

        let mut variables = rdf_results.variables().to_vec();
        for binding in &ts_bindings {
            for var in binding.bindings.keys() {
                if !variables.contains(var) {
                    variables.push(var.clone());
                }
            }
        }

        let merged = if rdf_bindings.is_empty() {
            ts_bindings
        } else {
            let mut joined = Vec::with_capacity(rdf_bindings.len() * ts_bindings.len());
            for rdf_row in rdf_bindings {
                for ts_row in &ts_bindings {
                    if bindings_are_compatible(rdf_row, ts_row) {
                        let mut combined = rdf_row.clone();
                        for (var, term) in &ts_row.bindings {
                            combined.bind(var.clone(), term.clone());
                        }
                        joined.push(combined);
                    }
                }
            }
            joined
        };

        Ok(OxirsQueryResults::from_bindings(merged, variables))
    }

    /// Add custom temporal function prefix
    pub fn add_temporal_prefix(&mut self, prefix: String) {
        if !self.temporal_prefixes.contains(&prefix) {
            self.temporal_prefixes.push(prefix);
        }
    }
}

impl Default for QueryRouter {
    fn default() -> Self {
        Self::new()
    }
}

/// SPARQL join compatibility check: two solution mappings are compatible if
/// every variable they have in common is bound to the same term.
fn bindings_are_compatible(a: &VariableBinding, b: &VariableBinding) -> bool {
    for (var, term) in &b.bindings {
        if let Some(existing) = a.bindings.get(var) {
            if existing != term {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxirs_core::RdfTerm;

    #[test]
    fn test_route_rdf_only() -> TsdbResult<()> {
        let router = QueryRouter::new();

        let query = r#"
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            SELECT ?s ?type WHERE {
                ?s rdf:type ?type .
            }
        "#;

        let decision = router.route(query)?;
        assert_eq!(decision, RoutingDecision::RdfOnly);

        Ok(())
    }

    #[test]
    fn test_route_timeseries_only() -> TsdbResult<()> {
        let router = QueryRouter::new();

        let query = r#"
            PREFIX qudt: <http://qudt.org/schema/qudt/>
            SELECT ?sensor ?value WHERE {
                ?sensor qudt:numericValue ?value .
            }
        "#;

        let decision = router.route(query)?;
        assert_eq!(decision, RoutingDecision::TimeseriesOnly);

        Ok(())
    }

    #[test]
    fn test_route_hybrid() -> TsdbResult<()> {
        let router = QueryRouter::new();

        let query = r#"
            PREFIX ts: <http://oxirs.org/ts#>
            PREFIX qudt: <http://qudt.org/schema/qudt/>
            SELECT ?sensor (ts:window(?value, 600, "AVG") AS ?avg) WHERE {
                ?sensor qudt:numericValue ?value .
            }
        "#;

        let decision = router.route(query)?;
        assert_eq!(decision, RoutingDecision::Hybrid);

        Ok(())
    }

    #[test]
    fn test_route_mixed_predicates() -> TsdbResult<()> {
        let router = QueryRouter::new();

        let query = r#"
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX qudt: <http://qudt.org/schema/qudt/>
            SELECT ?sensor ?type ?value WHERE {
                ?sensor rdf:type ?type ;
                        qudt:numericValue ?value .
            }
        "#;

        let decision = router.route(query)?;
        assert_eq!(decision, RoutingDecision::Hybrid);

        Ok(())
    }

    #[test]
    fn test_custom_temporal_prefix() -> TsdbResult<()> {
        let mut router = QueryRouter::new();
        router.add_temporal_prefix("custom:aggregate".to_string());

        let query = r#"
            SELECT ?sensor (custom:aggregate(?value) AS ?result) WHERE {
                ?sensor :value ?value .
            }
        "#;

        let decision = router.route(query)?;
        assert_eq!(decision, RoutingDecision::Hybrid);

        Ok(())
    }

    #[test]
    fn test_default_to_rdf() -> TsdbResult<()> {
        let router = QueryRouter::new();

        let query = r#"
            SELECT ?s ?p ?o WHERE {
                ?s ?p ?o .
            }
        "#;

        let decision = router.route(query)?;
        assert_eq!(decision, RoutingDecision::RdfOnly);

        Ok(())
    }

    // -- merge_results (P2 regression tests) --------------------------------

    fn ts_row(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    /// Regression test: time-series rows must actually appear in the merged
    /// result, not be discarded. See P2 finding on `QueryRouter::merge_results`.
    #[test]
    fn test_merge_results_joins_shared_variable() -> TsdbResult<()> {
        let router = QueryRouter::new();

        let mut sensor_binding = VariableBinding::new();
        sensor_binding.bind(
            "sensor".to_string(),
            Term::NamedNode(
                oxirs_core::model::NamedNode::new("http://example.org/sensor1").expect("valid IRI"),
            ),
        );
        let rdf_results =
            OxirsQueryResults::from_bindings(vec![sensor_binding], vec!["sensor".to_string()]);

        let ts_results = vec![ts_row(&[("value", "42.5")])];

        let merged = router.merge_results(rdf_results, ts_results)?;
        assert_eq!(merged.len(), 1, "expected exactly one merged row");
        assert!(merged.variables().contains(&"value".to_string()));

        if let QueryResults::Bindings(bindings) = merged.results() {
            let value = bindings[0]
                .get("value")
                .expect("value binding should be present");
            match value {
                Term::Literal(lit) => assert_eq!(lit.as_str(), "42.5"),
                other => panic!("expected literal, got {other:?}"),
            }
        } else {
            panic!("expected Bindings result");
        }

        Ok(())
    }

    #[test]
    fn test_merge_results_empty_ts_returns_rdf_unchanged() -> TsdbResult<()> {
        let router = QueryRouter::new();
        let rdf_results = OxirsQueryResults::from_bindings(Vec::new(), vec!["s".to_string()]);

        let merged = router.merge_results(rdf_results, Vec::new())?;
        assert!(merged.is_empty());

        Ok(())
    }

    #[test]
    fn test_merge_results_no_rdf_rows_falls_back_to_ts_rows() -> TsdbResult<()> {
        let router = QueryRouter::new();
        let rdf_results = OxirsQueryResults::from_bindings(Vec::new(), vec!["sensor".to_string()]);

        let ts_results = vec![ts_row(&[("value", "1.0")]), ts_row(&[("value", "2.0")])];

        let merged = router.merge_results(rdf_results, ts_results)?;
        assert_eq!(merged.len(), 2, "TS-only rows should pass through");

        Ok(())
    }

    #[test]
    fn test_merge_results_non_bindings_rdf_result_surfaces_ts_rows() -> TsdbResult<()> {
        let router = QueryRouter::new();
        let rdf_results = OxirsQueryResults::from_boolean(true);

        let ts_results = vec![ts_row(&[("value", "3.0")])];

        let merged = router.merge_results(rdf_results, ts_results)?;
        assert_eq!(merged.len(), 1);

        Ok(())
    }
}
