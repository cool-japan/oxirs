//! SPARQL Query Router for hybrid RDF + time-series queries
//!
//! This module routes SPARQL queries to the appropriate backend
//! and merges results from RDF and time-series storage.

use crate::error::TsdbResult;
use oxirs_core::rdf_store::OxirsQueryResults;
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
    pub fn merge_results(
        &self,
        rdf_results: OxirsQueryResults,
        _ts_results: Vec<HashMap<String, String>>,
    ) -> TsdbResult<OxirsQueryResults> {
        // For now, just return RDF results
        // In a full implementation, this would merge solution mappings
        Ok(rdf_results)
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
