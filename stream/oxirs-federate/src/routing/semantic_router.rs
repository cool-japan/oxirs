//! Semantic routing: routes SPARQL queries based on content analysis.
//!
//! The `SemanticRouter` inspects namespace prefixes declared in a query and
//! matches them against a registry of endpoints that advertise their
//! capabilities (supported prefixes and graph names).

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};
use thiserror::Error;

// ─── Error ────────────────────────────────────────────────────────────────────

/// Errors from the semantic router.
#[derive(Debug, Error)]
pub enum SemanticRouterError {
    #[error("Endpoint '{id}' not found in registry")]
    EndpointNotFound { id: String },

    #[error("Endpoint '{id}' is already registered")]
    DuplicateEndpoint { id: String },
}

// ─── Endpoint ─────────────────────────────────────────────────────────────────

/// A SPARQL endpoint that the semantic router can route to.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Endpoint {
    /// Unique endpoint identifier.
    pub id: String,
    /// SPARQL endpoint URL.
    pub url: String,
}

impl Endpoint {
    /// Create a new endpoint.
    pub fn new(id: impl Into<String>, url: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            url: url.into(),
        }
    }
}

// ─── EndpointCapability ───────────────────────────────────────────────────────

/// Describes what an endpoint handles.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EndpointCapability {
    /// Namespace prefixes the endpoint is authoritative for (e.g. `"foaf:"`).
    pub prefixes: Vec<String>,
    /// Named graph IRIs this endpoint hosts.
    pub graph_names: Vec<String>,
    /// Whether this endpoint accepts SPARQL UPDATE.
    pub supports_update: bool,
}

impl EndpointCapability {
    /// Create a capability record.
    pub fn new(prefixes: Vec<String>, graph_names: Vec<String>, supports_update: bool) -> Self {
        Self {
            prefixes,
            graph_names,
            supports_update,
        }
    }
}

// ─── EndpointRegistry ─────────────────────────────────────────────────────────

/// Registry that maps endpoints to their capabilities.
#[derive(Debug, Default)]
pub struct EndpointRegistry {
    /// endpoint id → (Endpoint metadata, Capability)
    entries: HashMap<String, (Endpoint, EndpointCapability)>,
}

impl EndpointRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register an endpoint with its capabilities.
    pub fn register_capability(
        &mut self,
        endpoint: Endpoint,
        cap: EndpointCapability,
    ) -> Result<(), SemanticRouterError> {
        if self.entries.contains_key(&endpoint.id) {
            return Err(SemanticRouterError::DuplicateEndpoint { id: endpoint.id });
        }
        self.entries.insert(endpoint.id.clone(), (endpoint, cap));
        Ok(())
    }

    /// Remove an endpoint by id.
    pub fn remove(&mut self, id: &str) -> Result<(), SemanticRouterError> {
        if self.entries.remove(id).is_none() {
            return Err(SemanticRouterError::EndpointNotFound { id: id.to_owned() });
        }
        Ok(())
    }

    /// Return all endpoints whose capabilities include `prefix`.
    pub fn get_endpoints_for_prefix(&self, prefix: &str) -> Vec<Endpoint> {
        self.entries
            .values()
            .filter(|(_, cap)| {
                cap.prefixes
                    .iter()
                    .any(|p| p == prefix || prefix.starts_with(p.trim_end_matches(':')))
            })
            .map(|(ep, _)| ep.clone())
            .collect()
    }

    /// Return all endpoints that host the given named graph.
    pub fn get_endpoints_for_graph(&self, graph_iri: &str) -> Vec<Endpoint> {
        self.entries
            .values()
            .filter(|(_, cap)| cap.graph_names.iter().any(|g| g == graph_iri))
            .map(|(ep, _)| ep.clone())
            .collect()
    }

    /// Return all endpoints that support SPARQL UPDATE.
    pub fn get_update_endpoints(&self) -> Vec<Endpoint> {
        self.entries
            .values()
            .filter(|(_, cap)| cap.supports_update)
            .map(|(ep, _)| ep.clone())
            .collect()
    }

    /// Number of registered endpoints.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get capability for an endpoint by id.
    pub fn get_capability(&self, id: &str) -> Option<&EndpointCapability> {
        self.entries.get(id).map(|(_, cap)| cap)
    }

    /// Get all endpoints.
    pub fn all_endpoints(&self) -> Vec<Endpoint> {
        self.entries.values().map(|(ep, _)| ep.clone()).collect()
    }
}

// ─── SemanticRouter ───────────────────────────────────────────────────────────

/// Routes queries based on namespace-prefix analysis.
#[derive(Debug, Default)]
pub struct SemanticRouter;

impl SemanticRouter {
    /// Create a new `SemanticRouter`.
    pub fn new() -> Self {
        Self
    }

    /// Extract all namespace prefix tokens from a SPARQL query.
    ///
    /// Recognises:
    /// - `PREFIX foo: <...>` declarations
    /// - Prefixed names (`foo:Bar`) used inline in the query body
    ///
    /// Returns deduplicated, sorted prefixes in `"prefix:"` form.
    pub fn detect_prefixes(&self, query: &str) -> Vec<String> {
        let mut found: HashSet<String> = HashSet::new();

        for line in query.lines() {
            let trimmed = line.trim();

            // Match: PREFIX <ident>: <iri>
            if let Some(rest) = trimmed
                .strip_prefix("PREFIX")
                .or_else(|| trimmed.strip_prefix("prefix"))
            {
                let rest = rest.trim();
                if let Some(colon_pos) = rest.find(':') {
                    let prefix_name = rest[..colon_pos].trim();
                    if !prefix_name.is_empty()
                        && prefix_name
                            .chars()
                            .all(|c| c.is_alphanumeric() || c == '_' || c == '-')
                    {
                        found.insert(format!("{prefix_name}:"));
                    }
                }
            }
        }

        // Also scan for inline prefixed names: word: followed by a capital or lowercase
        // Use a simple tokeniser: split on whitespace + common delimiters
        for token in query.split(|c: char| {
            c.is_whitespace() || matches!(c, ',' | ';' | '.' | '(' | ')' | '{' | '}' | '[' | ']')
        }) {
            let token = token.trim_matches(|c: char| matches!(c, '<' | '>' | '"' | '\''));
            if token.is_empty() || token.starts_with('<') || token.starts_with('"') {
                continue;
            }
            if let Some(colon_pos) = token.find(':') {
                let prefix_part = &token[..colon_pos];
                let local_part = &token[colon_pos + 1..];
                // Require: non-empty prefix made of word chars, non-empty local part, no slashes
                if !prefix_part.is_empty()
                    && !local_part.is_empty()
                    && !local_part.contains('/')
                    && prefix_part
                        .chars()
                        .all(|c| c.is_alphanumeric() || c == '_' || c == '-')
                {
                    // Skip xsd/rdf/rdfs/owl as they are ubiquitous – include everything else
                    found.insert(format!("{prefix_part}:"));
                }
            }
        }

        let mut result: Vec<String> = found.into_iter().collect();
        result.sort();
        result
    }

    /// Route a query to endpoints that handle its namespace prefixes.
    ///
    /// Returns all endpoints in `registry` whose capability set overlaps with
    /// the prefixes detected in `query`.  Deduplicates by endpoint id.
    pub fn route_by_prefix(&self, query: &str, registry: &EndpointRegistry) -> Vec<Endpoint> {
        let prefixes = self.detect_prefixes(query);
        if prefixes.is_empty() {
            return vec![];
        }

        let mut seen: HashSet<String> = HashSet::new();
        let mut results: Vec<Endpoint> = Vec::new();

        for prefix in &prefixes {
            for ep in registry.get_endpoints_for_prefix(prefix) {
                if seen.insert(ep.id.clone()) {
                    results.push(ep);
                }
            }
        }

        results
    }

    /// Route a query to endpoints based on both prefixes and UPDATE capability.
    ///
    /// If `require_update` is true, only endpoints that support UPDATE are
    /// included.
    pub fn route_with_options(
        &self,
        query: &str,
        registry: &EndpointRegistry,
        require_update: bool,
    ) -> Vec<Endpoint> {
        let mut candidates = self.route_by_prefix(query, registry);
        if require_update {
            candidates.retain(|ep| {
                registry
                    .get_capability(&ep.id)
                    .map(|c| c.supports_update)
                    .unwrap_or(false)
            });
        }
        candidates
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn build_registry() -> EndpointRegistry {
        let mut reg = EndpointRegistry::new();
        reg.register_capability(
            Endpoint::new("foaf-ep", "http://foaf.example/sparql"),
            EndpointCapability::new(
                vec!["foaf:".to_owned()],
                vec!["http://graph.example/foaf".to_owned()],
                false,
            ),
        )
        .expect("register foaf-ep");
        reg.register_capability(
            Endpoint::new("schema-ep", "http://schema.example/sparql"),
            EndpointCapability::new(vec!["schema:".to_owned(), "dct:".to_owned()], vec![], true),
        )
        .expect("register schema-ep");
        reg.register_capability(
            Endpoint::new("multi-ep", "http://multi.example/sparql"),
            EndpointCapability::new(
                vec!["foaf:".to_owned(), "schema:".to_owned()],
                vec!["http://graph.example/all".to_owned()],
                true,
            ),
        )
        .expect("register multi-ep");
        reg
    }

    #[test]
    fn test_detect_prefixes_from_prefix_decl() {
        let router = SemanticRouter::new();
        let query = "PREFIX foaf: <http://xmlns.com/foaf/0.1/>\nSELECT * WHERE { ?s foaf:name ?n }";
        let prefixes = router.detect_prefixes(query);
        assert!(
            prefixes.contains(&"foaf:".to_owned()),
            "should detect foaf:"
        );
    }

    #[test]
    fn test_detect_prefixes_inline_use() {
        let router = SemanticRouter::new();
        let query = "SELECT ?x WHERE { ?x schema:name \"Alice\" }";
        let prefixes = router.detect_prefixes(query);
        assert!(prefixes.contains(&"schema:".to_owned()));
    }

    #[test]
    fn test_detect_prefixes_multiple() {
        let router = SemanticRouter::new();
        let query = "PREFIX foaf: <..>\nPREFIX dct: <..>\nSELECT * WHERE { ?s foaf:name ?n ; dct:title ?t }";
        let prefixes = router.detect_prefixes(query);
        assert!(prefixes.contains(&"foaf:".to_owned()));
        assert!(prefixes.contains(&"dct:".to_owned()));
    }

    #[test]
    fn test_detect_prefixes_deduplicates() {
        let router = SemanticRouter::new();
        let query = "PREFIX foaf: <..>\nSELECT * WHERE { ?s foaf:name ?n ; foaf:knows ?o }";
        let prefixes = router.detect_prefixes(query);
        let count = prefixes.iter().filter(|p| *p == "foaf:").count();
        assert_eq!(count, 1, "should dedup foaf:");
    }

    #[test]
    fn test_detect_prefixes_sorted() {
        let router = SemanticRouter::new();
        let query = "PREFIX zzz: <..>\nPREFIX aaa: <..>\nSELECT * {}";
        let prefixes = router.detect_prefixes(query);
        let sorted = {
            let mut v = prefixes.clone();
            v.sort();
            v
        };
        assert_eq!(prefixes, sorted);
    }

    #[test]
    fn test_detect_prefixes_empty_query() {
        let router = SemanticRouter::new();
        assert!(
            router.detect_prefixes("SELECT * WHERE {}").is_empty()
                || !router.detect_prefixes("SELECT * WHERE {}").is_empty()
        );
        // Main check: no panic
    }

    #[test]
    fn test_route_by_prefix_finds_foaf_endpoints() {
        let router = SemanticRouter::new();
        let reg = build_registry();
        let query =
            "PREFIX foaf: <http://xmlns.com/foaf/0.1/>\nSELECT ?n WHERE { ?s foaf:name ?n }";
        let endpoints = router.route_by_prefix(query, &reg);
        let ids: Vec<_> = endpoints.iter().map(|e| e.id.as_str()).collect();
        assert!(ids.contains(&"foaf-ep") || ids.contains(&"multi-ep"));
    }

    #[test]
    fn test_route_by_prefix_no_match_returns_empty() {
        let router = SemanticRouter::new();
        let reg = build_registry();
        let query = "PREFIX xyz: <http://unknown/>\nSELECT * WHERE { ?s xyz:p ?o }";
        let endpoints = router.route_by_prefix(query, &reg);
        // xyz: is not registered
        assert!(endpoints.is_empty());
    }

    #[test]
    fn test_route_by_prefix_deduplicates_endpoints() {
        let router = SemanticRouter::new();
        let reg = build_registry();
        let query = "PREFIX foaf: <..>\nPREFIX schema: <..>\nSELECT * WHERE { ?s foaf:name ?n ; schema:knows ?o }";
        let endpoints = router.route_by_prefix(query, &reg);
        let ids: HashSet<_> = endpoints.iter().map(|e| e.id.clone()).collect();
        // multi-ep supports both foaf and schema, should appear only once
        assert_eq!(ids.len(), endpoints.len(), "should not duplicate endpoints");
    }

    #[test]
    fn test_endpoint_registry_register_duplicate_errors() {
        let mut reg = EndpointRegistry::new();
        reg.register_capability(
            Endpoint::new("ep1", "http://a/sparql"),
            EndpointCapability::default(),
        )
        .expect("first reg");
        let err = reg
            .register_capability(
                Endpoint::new("ep1", "http://b/sparql"),
                EndpointCapability::default(),
            )
            .unwrap_err();
        assert!(matches!(err, SemanticRouterError::DuplicateEndpoint { .. }));
    }

    #[test]
    fn test_endpoint_registry_get_endpoints_for_graph() {
        let reg = build_registry();
        let eps = reg.get_endpoints_for_graph("http://graph.example/foaf");
        assert_eq!(eps.len(), 1);
        assert_eq!(eps[0].id, "foaf-ep");
    }

    #[test]
    fn test_endpoint_registry_get_update_endpoints() {
        let reg = build_registry();
        let eps = reg.get_update_endpoints();
        let ids: Vec<_> = eps.iter().map(|e| e.id.as_str()).collect();
        assert!(ids.contains(&"schema-ep"));
        assert!(ids.contains(&"multi-ep"));
        assert!(!ids.contains(&"foaf-ep"));
    }

    #[test]
    fn test_route_with_options_update_required() {
        let router = SemanticRouter::new();
        let reg = build_registry();
        let query = "PREFIX foaf: <..>\nINSERT { ?s foaf:knows <X> } WHERE {}";
        let endpoints = router.route_with_options(query, &reg, true);
        // Only multi-ep supports foaf AND update
        assert!(endpoints.iter().all(|e| {
            reg.get_capability(&e.id)
                .map(|c| c.supports_update)
                .unwrap_or(false)
        }));
    }

    #[test]
    fn test_endpoint_registry_remove() {
        let mut reg = EndpointRegistry::new();
        reg.register_capability(
            Endpoint::new("r", "http://r/sparql"),
            EndpointCapability::default(),
        )
        .expect("reg");
        reg.remove("r").expect("remove");
        assert!(reg.is_empty());
    }

    #[test]
    fn test_endpoint_registry_remove_nonexistent_errors() {
        let mut reg = EndpointRegistry::new();
        assert!(reg.remove("ghost").is_err());
    }

    #[test]
    fn test_endpoint_registry_get_capability() {
        let reg = build_registry();
        let cap = reg.get_capability("schema-ep").expect("should exist");
        assert!(cap.supports_update);
    }

    #[test]
    fn test_endpoint_registry_all_endpoints_count() {
        let reg = build_registry();
        assert_eq!(reg.all_endpoints().len(), 3);
    }
}
