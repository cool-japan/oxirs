/// SPARQL 1.1 SERVICE clause for federated queries.
///
/// Implements the SERVICE clause enabling SPARQL queries to include
/// results from remote SPARQL endpoints.
use std::collections::HashMap;

/// A remote SPARQL endpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServiceEndpoint {
    /// The IRI of the endpoint.
    pub iri: String,
    /// Optional human-readable label.
    pub label: Option<String>,
}

/// A SPARQL SERVICE clause pattern binding a set of triple patterns to an endpoint.
#[derive(Debug, Clone)]
pub struct ServicePattern {
    /// The target endpoint.
    pub endpoint: ServiceEndpoint,
    /// Triple patterns to send to the endpoint.
    pub patterns: Vec<TriplePattern>,
    /// If true, errors from the endpoint are silently ignored.
    pub silent: bool,
}

/// A single triple pattern (subject, predicate, object).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TriplePattern {
    /// Subject (IRI, variable, or literal).
    pub s: String,
    /// Predicate.
    pub p: String,
    /// Object.
    pub o: String,
}

/// A single variable binding returned from a remote service call.
#[derive(Debug, Clone)]
pub struct ServiceBinding {
    /// The endpoint IRI that produced this binding.
    pub endpoint: String,
    /// The variable name.
    pub var: String,
    /// The bound value.
    pub value: String,
}

/// The result of executing a SERVICE pattern.
#[derive(Debug, Clone)]
pub struct ServiceResult {
    /// Row-based bindings (each HashMap is one solution row).
    pub bindings: Vec<HashMap<String, String>>,
    /// The endpoint IRI.
    pub endpoint: String,
    /// Simulated elapsed time in milliseconds.
    pub elapsed_ms: u64,
    /// Non-None when the endpoint call failed (and silent=false means propagate).
    pub error: Option<String>,
}

/// Processor for SPARQL 1.1 SERVICE clauses.
#[derive(Debug, Default)]
pub struct ServiceClauseProcessor {
    endpoints: HashMap<String, ServiceEndpoint>,
}

impl ServiceClauseProcessor {
    /// Create a new, empty processor.
    pub fn new() -> Self {
        Self {
            endpoints: HashMap::new(),
        }
    }

    /// Register a remote endpoint so it can be referenced by IRI.
    pub fn register_endpoint(&mut self, endpoint: ServiceEndpoint) {
        self.endpoints.insert(endpoint.iri.clone(), endpoint);
    }

    /// Look up a registered endpoint by IRI.
    pub fn get_endpoint(&self, iri: &str) -> Option<&ServiceEndpoint> {
        self.endpoints.get(iri)
    }

    /// Validate a SERVICE pattern: endpoint must be registered, patterns must be non-trivially formed.
    pub fn validate_pattern(&self, pattern: &ServicePattern) -> Result<(), ServiceError> {
        if !self.endpoints.contains_key(&pattern.endpoint.iri) {
            return Err(ServiceError::UnknownEndpoint(pattern.endpoint.iri.clone()));
        }
        for tp in &pattern.patterns {
            if tp.s.is_empty() || tp.p.is_empty() || tp.o.is_empty() {
                return Err(ServiceError::InvalidPattern(format!(
                    "Triple pattern has empty component: ({}, {}, {})",
                    tp.s, tp.p, tp.o
                )));
            }
        }
        Ok(())
    }

    /// Simulate executing a SERVICE pattern against the remote endpoint.
    ///
    /// In a real implementation this would issue an HTTP SPARQL request.
    /// Here we generate synthetic bindings based on the patterns and input bindings.
    pub fn simulate_execute(
        &self,
        pattern: &ServicePattern,
        input_bindings: &[HashMap<String, String>],
    ) -> ServiceResult {
        let endpoint_iri = pattern.endpoint.iri.clone();

        // If the endpoint is not registered, return an error result.
        if !self.endpoints.contains_key(&endpoint_iri) {
            return ServiceResult {
                bindings: vec![],
                endpoint: endpoint_iri,
                elapsed_ms: 0,
                error: Some(format!("Unknown endpoint: {}", pattern.endpoint.iri)),
            };
        }

        // Simulate: for each pattern generate one synthetic binding row.
        // Variables start with '?', constants are passed through.
        let start = std::time::Instant::now();

        let base_bindings: Vec<HashMap<String, String>> = if input_bindings.is_empty() {
            vec![HashMap::new()]
        } else {
            input_bindings.to_vec()
        };

        let mut result_bindings = Vec::new();
        for input_row in &base_bindings {
            for (idx, tp) in pattern.patterns.iter().enumerate() {
                let mut row = input_row.clone();
                // Bind variables in the triple pattern to simulated values.
                if tp.s.starts_with('?') {
                    row.entry(tp.s[1..].to_string())
                        .or_insert_with(|| format!("sim_s_{idx}"));
                }
                if tp.p.starts_with('?') {
                    row.entry(tp.p[1..].to_string())
                        .or_insert_with(|| format!("sim_p_{idx}"));
                }
                if tp.o.starts_with('?') {
                    row.entry(tp.o[1..].to_string())
                        .or_insert_with(|| format!("sim_o_{idx}"));
                }
                result_bindings.push(row);
            }
        }

        // Deduplicate identical rows.
        let mut seen = std::collections::HashSet::new();
        result_bindings.retain(|row| {
            let key: Vec<(&String, &String)> = {
                let mut pairs: Vec<(&String, &String)> = row.iter().collect();
                pairs.sort_by_key(|(k, _)| k.as_str());
                pairs
            };
            let key_str = format!("{key:?}");
            seen.insert(key_str)
        });

        let elapsed_ms = start.elapsed().as_millis() as u64;

        ServiceResult {
            bindings: result_bindings,
            endpoint: endpoint_iri,
            elapsed_ms,
            error: None,
        }
    }

    /// Return the count of registered endpoints.
    pub fn endpoint_count(&self) -> usize {
        self.endpoints.len()
    }

    /// Merge local bindings with remote SERVICE results (natural join on shared variables).
    pub fn merge_results(
        local: &[HashMap<String, String>],
        remote: &ServiceResult,
    ) -> Vec<HashMap<String, String>> {
        if remote.error.is_some() {
            // Error result: return empty (silent mode handled by caller)
            return vec![];
        }

        if local.is_empty() {
            return remote.bindings.clone();
        }

        if remote.bindings.is_empty() {
            return vec![];
        }

        let mut merged = Vec::new();
        for local_row in local {
            for remote_row in &remote.bindings {
                // Check compatibility: shared variables must agree.
                let compatible = remote_row
                    .iter()
                    .all(|(k, v)| local_row.get(k).map_or(true, |lv| lv == v));
                if compatible {
                    let mut combined = local_row.clone();
                    for (k, v) in remote_row {
                        combined.entry(k.clone()).or_insert_with(|| v.clone());
                    }
                    merged.push(combined);
                }
            }
        }
        merged
    }

    /// Return whether a pattern uses the SILENT modifier.
    pub fn is_silent(&self, pattern: &ServicePattern) -> bool {
        pattern.silent
    }
}

/// Errors that can occur when processing SERVICE clauses.
#[derive(Debug)]
pub enum ServiceError {
    /// The endpoint IRI was not registered.
    UnknownEndpoint(String),
    /// The SERVICE pattern is structurally invalid.
    InvalidPattern(String),
    /// The remote execution failed.
    ExecutionFailed(String),
}

impl std::fmt::Display for ServiceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownEndpoint(iri) => write!(f, "Unknown endpoint: {iri}"),
            Self::InvalidPattern(msg) => write!(f, "Invalid pattern: {msg}"),
            Self::ExecutionFailed(msg) => write!(f, "Execution failed: {msg}"),
        }
    }
}

impl std::error::Error for ServiceError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_endpoint(iri: &str, label: Option<&str>) -> ServiceEndpoint {
        ServiceEndpoint {
            iri: iri.to_string(),
            label: label.map(|s| s.to_string()),
        }
    }

    fn make_pattern(iri: &str, silent: bool) -> ServicePattern {
        ServicePattern {
            endpoint: make_endpoint(iri, None),
            patterns: vec![TriplePattern {
                s: "?s".to_string(),
                p: "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string(),
                o: "?o".to_string(),
            }],
            silent,
        }
    }

    // --- endpoint registration ---

    #[test]
    fn test_register_endpoint() {
        let mut proc = ServiceClauseProcessor::new();
        proc.register_endpoint(make_endpoint("http://dbpedia.org/sparql", Some("DBpedia")));
        assert!(proc.get_endpoint("http://dbpedia.org/sparql").is_some());
    }

    #[test]
    fn test_get_endpoint_label() {
        let mut proc = ServiceClauseProcessor::new();
        proc.register_endpoint(make_endpoint("http://example.org/sparql", Some("Example")));
        let ep = proc.get_endpoint("http://example.org/sparql").unwrap();
        assert_eq!(ep.label.as_deref(), Some("Example"));
    }

    #[test]
    fn test_get_endpoint_none_label() {
        let mut proc = ServiceClauseProcessor::new();
        proc.register_endpoint(make_endpoint("http://example.org/sparql", None));
        let ep = proc.get_endpoint("http://example.org/sparql").unwrap();
        assert!(ep.label.is_none());
    }

    #[test]
    fn test_get_unknown_endpoint() {
        let proc = ServiceClauseProcessor::new();
        assert!(proc.get_endpoint("http://nowhere.org/sparql").is_none());
    }

    #[test]
    fn test_endpoint_count_zero() {
        let proc = ServiceClauseProcessor::new();
        assert_eq!(proc.endpoint_count(), 0);
    }

    #[test]
    fn test_endpoint_count_one() {
        let mut proc = ServiceClauseProcessor::new();
        proc.register_endpoint(make_endpoint("http://a.org/sparql", None));
        assert_eq!(proc.endpoint_count(), 1);
    }

    #[test]
    fn test_endpoint_count_multiple() {
        let mut proc = ServiceClauseProcessor::new();
        proc.register_endpoint(make_endpoint("http://a.org/sparql", None));
        proc.register_endpoint(make_endpoint("http://b.org/sparql", None));
        proc.register_endpoint(make_endpoint("http://c.org/sparql", None));
        assert_eq!(proc.endpoint_count(), 3);
    }

    #[test]
    fn test_register_overwrites() {
        let mut proc = ServiceClauseProcessor::new();
        proc.register_endpoint(make_endpoint("http://ep.org/sparql", Some("v1")));
        proc.register_endpoint(make_endpoint("http://ep.org/sparql", Some("v2")));
        assert_eq!(proc.endpoint_count(), 1);
        assert_eq!(
            proc.get_endpoint("http://ep.org/sparql")
                .unwrap()
                .label
                .as_deref(),
            Some("v2")
        );
    }

    // --- validate_pattern ---

    #[test]
    fn test_validate_known_endpoint() {
        let mut proc = ServiceClauseProcessor::new();
        proc.register_endpoint(make_endpoint("http://ep.org/sparql", None));
        let pat = make_pattern("http://ep.org/sparql", false);
        assert!(proc.validate_pattern(&pat).is_ok());
    }

    #[test]
    fn test_validate_unknown_endpoint() {
        let proc = ServiceClauseProcessor::new();
        let pat = make_pattern("http://unknown.org/sparql", false);
        match proc.validate_pattern(&pat) {
            Err(ServiceError::UnknownEndpoint(_)) => {}
            _ => panic!("Expected UnknownEndpoint error"),
        }
    }

    #[test]
    fn test_validate_empty_subject() {
        let mut proc = ServiceClauseProcessor::new();
        proc.register_endpoint(make_endpoint("http://ep.org/sparql", None));
        let pat = ServicePattern {
            endpoint: make_endpoint("http://ep.org/sparql", None),
            patterns: vec![TriplePattern {
                s: "".to_string(),
                p: "http://p".to_string(),
                o: "http://o".to_string(),
            }],
            silent: false,
        };
        match proc.validate_pattern(&pat) {
            Err(ServiceError::InvalidPattern(_)) => {}
            _ => panic!("Expected InvalidPattern"),
        }
    }

    #[test]
    fn test_validate_empty_predicate() {
        let mut proc = ServiceClauseProcessor::new();
        proc.register_endpoint(make_endpoint("http://ep.org/sparql", None));
        let pat = ServicePattern {
            endpoint: make_endpoint("http://ep.org/sparql", None),
            patterns: vec![TriplePattern {
                s: "?s".to_string(),
                p: "".to_string(),
                o: "?o".to_string(),
            }],
            silent: false,
        };
        assert!(proc.validate_pattern(&pat).is_err());
    }

    #[test]
    fn test_validate_empty_object() {
        let mut proc = ServiceClauseProcessor::new();
        proc.register_endpoint(make_endpoint("http://ep.org/sparql", None));
        let pat = ServicePattern {
            endpoint: make_endpoint("http://ep.org/sparql", None),
            patterns: vec![TriplePattern {
                s: "?s".to_string(),
                p: "http://p".to_string(),
                o: "".to_string(),
            }],
            silent: false,
        };
        assert!(proc.validate_pattern(&pat).is_err());
    }

    #[test]
    fn test_validate_empty_patterns_list() {
        let mut proc = ServiceClauseProcessor::new();
        proc.register_endpoint(make_endpoint("http://ep.org/sparql", None));
        let pat = ServicePattern {
            endpoint: make_endpoint("http://ep.org/sparql", None),
            patterns: vec![],
            silent: false,
        };
        // Empty patterns list is valid (no patterns to check)
        assert!(proc.validate_pattern(&pat).is_ok());
    }

    // --- simulate_execute ---

    #[test]
    fn test_simulate_execute_returns_bindings() {
        let mut proc = ServiceClauseProcessor::new();
        proc.register_endpoint(make_endpoint("http://ep.org/sparql", None));
        let pat = make_pattern("http://ep.org/sparql", false);
        let result = proc.simulate_execute(&pat, &[]);
        assert!(result.error.is_none());
        assert!(!result.bindings.is_empty());
    }

    #[test]
    fn test_simulate_execute_endpoint_in_result() {
        let mut proc = ServiceClauseProcessor::new();
        proc.register_endpoint(make_endpoint("http://ep.org/sparql", None));
        let pat = make_pattern("http://ep.org/sparql", false);
        let result = proc.simulate_execute(&pat, &[]);
        assert_eq!(result.endpoint, "http://ep.org/sparql");
    }

    #[test]
    fn test_simulate_execute_unknown_returns_error() {
        let proc = ServiceClauseProcessor::new();
        let pat = make_pattern("http://unknown.org/sparql", false);
        let result = proc.simulate_execute(&pat, &[]);
        assert!(result.error.is_some());
        assert!(result.bindings.is_empty());
    }

    #[test]
    fn test_simulate_execute_binds_variables() {
        let mut proc = ServiceClauseProcessor::new();
        proc.register_endpoint(make_endpoint("http://ep.org/sparql", None));
        let pat = make_pattern("http://ep.org/sparql", false);
        let result = proc.simulate_execute(&pat, &[]);
        // All bindings should have "s" and "o" (the variables in the pattern).
        for row in &result.bindings {
            assert!(row.contains_key("s") || row.contains_key("o"));
        }
    }

    #[test]
    fn test_simulate_execute_with_input_bindings() {
        let mut proc = ServiceClauseProcessor::new();
        proc.register_endpoint(make_endpoint("http://ep.org/sparql", None));
        let pat = make_pattern("http://ep.org/sparql", false);
        let mut input = HashMap::new();
        input.insert("existing".to_string(), "val1".to_string());
        let result = proc.simulate_execute(&pat, &[input]);
        assert!(result.error.is_none());
        for row in &result.bindings {
            assert_eq!(row.get("existing").map(|s| s.as_str()), Some("val1"));
        }
    }

    #[test]
    fn test_simulate_execute_elapsed_ms() {
        let mut proc = ServiceClauseProcessor::new();
        proc.register_endpoint(make_endpoint("http://ep.org/sparql", None));
        let pat = make_pattern("http://ep.org/sparql", false);
        let result = proc.simulate_execute(&pat, &[]);
        // elapsed_ms should be a reasonable number (not usize::MAX etc.)
        assert!(result.elapsed_ms < 10_000);
    }

    // --- silent mode ---

    #[test]
    fn test_is_silent_true() {
        let pat = make_pattern("http://ep.org/sparql", true);
        let proc = ServiceClauseProcessor::new();
        assert!(proc.is_silent(&pat));
    }

    #[test]
    fn test_is_silent_false() {
        let pat = make_pattern("http://ep.org/sparql", false);
        let proc = ServiceClauseProcessor::new();
        assert!(!proc.is_silent(&pat));
    }

    #[test]
    fn test_silent_mode_error_result() {
        let proc = ServiceClauseProcessor::new();
        let pat = make_pattern("http://unknown.org/sparql", true);
        let result = proc.simulate_execute(&pat, &[]);
        // Silent pattern should still get the error flag set, but caller ignores it
        assert!(result.error.is_some());
    }

    // --- merge_results ---

    #[test]
    fn test_merge_results_empty_local() {
        let remote = ServiceResult {
            bindings: vec![{
                let mut m = HashMap::new();
                m.insert("x".to_string(), "1".to_string());
                m
            }],
            endpoint: "http://ep.org/sparql".to_string(),
            elapsed_ms: 0,
            error: None,
        };
        let merged = ServiceClauseProcessor::merge_results(&[], &remote);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].get("x").map(|s| s.as_str()), Some("1"));
    }

    #[test]
    fn test_merge_results_empty_remote() {
        let local = vec![{
            let mut m = HashMap::new();
            m.insert("y".to_string(), "2".to_string());
            m
        }];
        let remote = ServiceResult {
            bindings: vec![],
            endpoint: "http://ep.org/sparql".to_string(),
            elapsed_ms: 0,
            error: None,
        };
        let merged = ServiceClauseProcessor::merge_results(&local, &remote);
        assert!(merged.is_empty());
    }

    #[test]
    fn test_merge_results_natural_join() {
        let local = vec![{
            let mut m = HashMap::new();
            m.insert("x".to_string(), "1".to_string());
            m
        }];
        let remote = ServiceResult {
            bindings: vec![{
                let mut m = HashMap::new();
                m.insert("x".to_string(), "1".to_string());
                m.insert("y".to_string(), "hello".to_string());
                m
            }],
            endpoint: "http://ep.org/sparql".to_string(),
            elapsed_ms: 0,
            error: None,
        };
        let merged = ServiceClauseProcessor::merge_results(&local, &remote);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].get("y").map(|s| s.as_str()), Some("hello"));
    }

    #[test]
    fn test_merge_results_incompatible_join() {
        let local = vec![{
            let mut m = HashMap::new();
            m.insert("x".to_string(), "1".to_string());
            m
        }];
        let remote = ServiceResult {
            bindings: vec![{
                let mut m = HashMap::new();
                m.insert("x".to_string(), "2".to_string()); // conflict
                m.insert("y".to_string(), "hello".to_string());
                m
            }],
            endpoint: "http://ep.org/sparql".to_string(),
            elapsed_ms: 0,
            error: None,
        };
        let merged = ServiceClauseProcessor::merge_results(&local, &remote);
        assert!(merged.is_empty());
    }

    #[test]
    fn test_merge_results_with_error() {
        let local = vec![{
            let mut m = HashMap::new();
            m.insert("x".to_string(), "1".to_string());
            m
        }];
        let remote = ServiceResult {
            bindings: vec![],
            endpoint: "http://ep.org/sparql".to_string(),
            elapsed_ms: 0,
            error: Some("timeout".to_string()),
        };
        let merged = ServiceClauseProcessor::merge_results(&local, &remote);
        assert!(merged.is_empty());
    }

    // --- multiple endpoints ---

    #[test]
    fn test_multiple_endpoints_registered() {
        let mut proc = ServiceClauseProcessor::new();
        let iris = [
            "http://a.org/sparql",
            "http://b.org/sparql",
            "http://c.org/sparql",
        ];
        for iri in &iris {
            proc.register_endpoint(make_endpoint(iri, None));
        }
        assert_eq!(proc.endpoint_count(), 3);
        for iri in &iris {
            assert!(proc.get_endpoint(iri).is_some());
        }
    }

    #[test]
    fn test_execute_multiple_patterns() {
        let mut proc = ServiceClauseProcessor::new();
        proc.register_endpoint(make_endpoint("http://ep.org/sparql", None));
        let pat = ServicePattern {
            endpoint: make_endpoint("http://ep.org/sparql", None),
            patterns: vec![
                TriplePattern {
                    s: "?s".to_string(),
                    p: "http://schema.org/name".to_string(),
                    o: "?name".to_string(),
                },
                TriplePattern {
                    s: "?s".to_string(),
                    p: "http://schema.org/age".to_string(),
                    o: "?age".to_string(),
                },
            ],
            silent: false,
        };
        let result = proc.simulate_execute(&pat, &[]);
        assert!(result.error.is_none());
        // Should have bindings for both patterns
        assert!(result.bindings.len() >= 2);
    }

    // --- ServiceError display ---

    #[test]
    fn test_unknown_endpoint_error_display() {
        let err = ServiceError::UnknownEndpoint("http://x.org/sparql".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("http://x.org/sparql"));
    }

    #[test]
    fn test_invalid_pattern_error_display() {
        let err = ServiceError::InvalidPattern("bad pattern".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("bad pattern"));
    }

    #[test]
    fn test_execution_failed_error_display() {
        let err = ServiceError::ExecutionFailed("timeout".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("timeout"));
    }

    // --- TriplePattern equality ---

    #[test]
    fn test_triple_pattern_equality() {
        let tp1 = TriplePattern {
            s: "?s".to_string(),
            p: "http://p".to_string(),
            o: "?o".to_string(),
        };
        let tp2 = TriplePattern {
            s: "?s".to_string(),
            p: "http://p".to_string(),
            o: "?o".to_string(),
        };
        assert_eq!(tp1, tp2);
    }

    #[test]
    fn test_service_processor_default() {
        let proc = ServiceClauseProcessor::default();
        assert_eq!(proc.endpoint_count(), 0);
    }

    #[test]
    fn test_merge_results_both_empty() {
        let remote = ServiceResult {
            bindings: vec![],
            endpoint: "http://ep.org/sparql".to_string(),
            elapsed_ms: 0,
            error: None,
        };
        let merged = ServiceClauseProcessor::merge_results(&[], &remote);
        assert!(merged.is_empty());
    }

    #[test]
    fn test_validate_multiple_patterns_all_valid() {
        let mut proc = ServiceClauseProcessor::new();
        proc.register_endpoint(make_endpoint("http://ep.org/sparql", None));
        let pat = ServicePattern {
            endpoint: make_endpoint("http://ep.org/sparql", None),
            patterns: vec![
                TriplePattern {
                    s: "?a".to_string(),
                    p: "http://p1".to_string(),
                    o: "?b".to_string(),
                },
                TriplePattern {
                    s: "?b".to_string(),
                    p: "http://p2".to_string(),
                    o: "?c".to_string(),
                },
            ],
            silent: false,
        };
        assert!(proc.validate_pattern(&pat).is_ok());
    }

    #[test]
    fn test_endpoint_iri_stored_correctly() {
        let mut proc = ServiceClauseProcessor::new();
        let ep = make_endpoint("http://wikidata.org/sparql", Some("Wikidata"));
        proc.register_endpoint(ep);
        let stored = proc.get_endpoint("http://wikidata.org/sparql").unwrap();
        assert_eq!(stored.iri, "http://wikidata.org/sparql");
        assert_eq!(stored.label.as_deref(), Some("Wikidata"));
    }
}
