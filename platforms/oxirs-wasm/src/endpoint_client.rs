//! # SPARQL Endpoint Client
//!
//! A simulated SPARQL 1.1 endpoint client suitable for testing and WASM environments
//! where real HTTP connections are unavailable.  The `execute` method builds a mock
//! JSON/boolean response based on the query type and tracks request statistics.
//!
//! ## Example
//!
//! ```rust
//! use oxirs_wasm::endpoint_client::{
//!     ClientConfig, EndpointClient, QueryType, SparqlQuery,
//! };
//!
//! let config = ClientConfig {
//!     endpoint_url: "http://localhost:3030/ds".to_string(),
//!     default_timeout_ms: 5000,
//!     max_retries: 3,
//!     accept_format: "application/sparql-results+json".to_string(),
//! };
//! let mut client = EndpointClient::new(config);
//!
//! let query = SparqlQuery {
//!     text: "SELECT ?s WHERE { ?s ?p ?o }".to_string(),
//!     query_type: QueryType::Select,
//!     timeout_ms: None,
//!     default_graph: None,
//! };
//! let response = client.execute(query).expect("execute failed");
//! assert!(response.result_json.contains("results"));
//! ```

/// SPARQL query type discriminator
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueryType {
    /// SPARQL SELECT — returns variable bindings
    Select,
    /// SPARQL ASK — returns a boolean
    Ask,
    /// SPARQL CONSTRUCT — returns a graph (triples)
    Construct,
    /// SPARQL DESCRIBE — returns a graph describing a resource
    Describe,
}

/// A SPARQL query to be executed
#[derive(Debug, Clone)]
pub struct SparqlQuery {
    /// The SPARQL query text
    pub text: String,
    /// Explicitly specified query type (use [`EndpointClient::detect_query_type`]
    /// to infer from text when unknown)
    pub query_type: QueryType,
    /// Per-query timeout override in milliseconds
    pub timeout_ms: Option<u64>,
    /// Default graph URI override
    pub default_graph: Option<String>,
}

/// Response from a SPARQL query execution
#[derive(Debug, Clone)]
pub struct QueryResponse {
    /// Query type that produced this response
    pub query_type: QueryType,
    /// JSON-encoded result body
    pub result_json: String,
    /// Simulated wall-clock time in milliseconds
    pub elapsed_ms: u64,
    /// Number of result rows / triples
    pub row_count: usize,
}

/// Endpoint client configuration
#[derive(Debug, Clone)]
pub struct ClientConfig {
    /// SPARQL endpoint URL (e.g. `"http://localhost:3030/ds/sparql"`)
    pub endpoint_url: String,
    /// Default request timeout in milliseconds
    pub default_timeout_ms: u64,
    /// Maximum number of retries on transient failure
    pub max_retries: usize,
    /// MIME type for the `Accept` header
    pub accept_format: String,
}

impl Default for ClientConfig {
    fn default() -> Self {
        Self {
            endpoint_url: "http://localhost:3030/sparql".to_string(),
            default_timeout_ms: 30_000,
            max_retries: 3,
            accept_format: "application/sparql-results+json".to_string(),
        }
    }
}

/// Aggregate request statistics
#[derive(Debug, Clone, Default)]
pub struct RequestStats {
    /// Total number of queries submitted (including updates)
    pub total_requests: u64,
    /// Queries that completed without error
    pub successful: u64,
    /// Queries that returned an error
    pub failed: u64,
    /// Cumulative elapsed time in milliseconds
    pub total_elapsed_ms: u64,
}

/// Errors returned by [`EndpointClient`]
#[derive(Debug)]
pub enum ClientError {
    /// Simulated network / connection failure
    NetworkError(String),
    /// Request timed out (timeout_ms exceeded)
    Timeout,
    /// Response body could not be parsed
    ParseError(String),
    /// The submitted SPARQL string is syntactically invalid
    InvalidQuery(String),
}

impl std::fmt::Display for ClientError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ClientError::NetworkError(msg) => write!(f, "network error: {}", msg),
            ClientError::Timeout => write!(f, "request timed out"),
            ClientError::ParseError(msg) => write!(f, "parse error: {}", msg),
            ClientError::InvalidQuery(msg) => write!(f, "invalid query: {}", msg),
        }
    }
}

impl std::error::Error for ClientError {}

// ─── Endpoint client ──────────────────────────────────────────────────────────

/// Simulated SPARQL endpoint client
pub struct EndpointClient {
    config: ClientConfig,
    stats: RequestStats,
    history: Vec<SparqlQuery>,
}

impl EndpointClient {
    /// Create a new client with the given configuration.
    pub fn new(config: ClientConfig) -> Self {
        Self {
            config,
            stats: RequestStats::default(),
            history: Vec::new(),
        }
    }

    // ── Query execution ─────────────────────────────────────────────────

    /// Execute a SPARQL query, returning a simulated [`QueryResponse`].
    ///
    /// The response is generated deterministically based on the query type:
    ///
    /// | Type      | Result                                             |
    /// |-----------|---------------------------------------------------|
    /// | SELECT    | SPARQL JSON results object with extracted vars     |
    /// | ASK       | SPARQL JSON boolean response                       |
    /// | CONSTRUCT | SPARQL JSON graph with a handful of mock triples   |
    /// | DESCRIBE  | Same as CONSTRUCT                                  |
    pub fn execute(&mut self, query: SparqlQuery) -> Result<QueryResponse, ClientError> {
        if query.text.trim().is_empty() {
            self.stats.total_requests += 1;
            self.stats.failed += 1;
            return Err(ClientError::InvalidQuery("query text is empty".to_string()));
        }

        // Simulate a timeout when the query contains the literal marker "__timeout__"
        if query.text.contains("__timeout__") {
            self.stats.total_requests += 1;
            self.stats.failed += 1;
            return Err(ClientError::Timeout);
        }

        let elapsed_ms: u64 = simulate_elapsed_ms(&query.text);

        let (result_json, row_count) = match &query.query_type {
            QueryType::Select => {
                let vars = extract_select_vars(&query.text);
                let var_refs: Vec<&str> = vars.iter().map(String::as_str).collect();
                let count = simulated_row_count(&query.text);
                let json = Self::build_select_response(&var_refs, count);
                (json, count)
            }
            QueryType::Ask => {
                let boolean = !query.text.to_lowercase().contains("false");
                let json = format!(
                    r#"{{"head":{{}}, "boolean":{}}}"#,
                    boolean
                );
                (json, if boolean { 1 } else { 0 })
            }
            QueryType::Construct | QueryType::Describe => {
                let count = 3usize;
                let json = build_graph_response(count);
                (json, count)
            }
        };

        self.history.push(query.clone());
        self.stats.total_requests += 1;
        self.stats.successful += 1;
        self.stats.total_elapsed_ms += elapsed_ms;

        Ok(QueryResponse {
            query_type: query.query_type,
            result_json,
            elapsed_ms,
            row_count,
        })
    }

    /// Execute a SPARQL Update (INSERT/DELETE/CLEAR/DROP/LOAD).
    ///
    /// Always succeeds in the simulation unless the string is empty.
    pub fn execute_update(&mut self, update: &str) -> Result<(), ClientError> {
        if update.trim().is_empty() {
            self.stats.total_requests += 1;
            self.stats.failed += 1;
            return Err(ClientError::InvalidQuery(
                "update string is empty".to_string(),
            ));
        }
        self.stats.total_requests += 1;
        self.stats.successful += 1;
        Ok(())
    }

    // ── Introspection ────────────────────────────────────────────────────

    /// Access accumulated request statistics.
    pub fn stats(&self) -> &RequestStats {
        &self.stats
    }

    /// Access the query history (most recent last).
    pub fn history(&self) -> &[SparqlQuery] {
        &self.history
    }

    /// Clear the query history.
    pub fn clear_history(&mut self) {
        self.history.clear();
    }

    /// The last query submitted, if any.
    pub fn last_query(&self) -> Option<&SparqlQuery> {
        self.history.last()
    }

    // ── Static helpers ───────────────────────────────────────────────────

    /// Detect the query type from the SPARQL text (case-insensitive keyword scan).
    ///
    /// Returns [`QueryType::Select`] when no other type keyword is found.
    pub fn detect_query_type(sparql: &str) -> QueryType {
        let upper = sparql.to_uppercase();
        // Check specific keyword presence in rough priority order
        if upper.contains("ASK") {
            return QueryType::Ask;
        }
        if upper.contains("CONSTRUCT") {
            return QueryType::Construct;
        }
        if upper.contains("DESCRIBE") {
            return QueryType::Describe;
        }
        QueryType::Select
    }

    /// Build a mock SPARQL JSON SELECT response for `vars` with `row_count` binding rows.
    ///
    /// Each variable gets a literal value of `"value_{var}_{row}"`.
    pub fn build_select_response(vars: &[&str], row_count: usize) -> String {
        let var_list: Vec<String> = vars.iter().map(|v| format!(r#""{}""#, v)).collect();
        let vars_json = format!("[{}]", var_list.join(", "));

        let mut bindings = Vec::new();
        for row in 0..row_count {
            let mut fields: Vec<String> = Vec::new();
            for var in vars {
                fields.push(format!(
                    r#""{}":{{"type":"literal","value":"value_{}_{}"}}"#,
                    var, var, row
                ));
            }
            bindings.push(format!("{{{}}}", fields.join(", ")));
        }

        format!(
            r#"{{"head":{{"vars":{vars}}},"results":{{"bindings":[{bindings}]}}}}"#,
            vars = vars_json,
            bindings = bindings.join(", ")
        )
    }
}

// ─── Private helpers ──────────────────────────────────────────────────────────

/// Extract `?var` names from a SELECT query text.
fn extract_select_vars(query: &str) -> Vec<String> {
    let mut vars = Vec::new();
    for token in query.split_whitespace() {
        if let Some(stripped) = token.strip_prefix('?') {
            let name: String = stripped
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            if !name.is_empty() && !vars.contains(&name) {
                vars.push(name);
            }
        }
    }
    if vars.is_empty() {
        vars.push("result".to_string());
    }
    vars
}

/// Deterministic simulated row count based on query text length.
fn simulated_row_count(query: &str) -> usize {
    let len = query.len();
    if len < 30 {
        1
    } else if len < 80 {
        3
    } else {
        5
    }
}

/// Deterministic simulated elapsed time (ms).
fn simulate_elapsed_ms(query: &str) -> u64 {
    (query.len() as u64 % 50) + 1
}

/// Build a minimal SPARQL JSON graph response (CONSTRUCT/DESCRIBE).
fn build_graph_response(triple_count: usize) -> String {
    let mut triples: Vec<String> = Vec::new();
    for i in 0..triple_count {
        triples.push(format!(
            r#"{{"subject":{{"type":"uri","value":"http://example.org/s{}"}},"predicate":{{"type":"uri","value":"http://example.org/p"}},"object":{{"type":"uri","value":"http://example.org/o{}"}}}}"#,
            i, i
        ));
    }
    format!(r#"{{"results":{{"triples":[{}]}}}}"#, triples.join(", "))
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_client() -> EndpointClient {
        EndpointClient::new(ClientConfig::default())
    }

    fn select_query(text: &str) -> SparqlQuery {
        SparqlQuery {
            text: text.to_string(),
            query_type: QueryType::Select,
            timeout_ms: None,
            default_graph: None,
        }
    }

    // ── execute SELECT ─────────────────────────────────────────────────────

    #[test]
    fn test_execute_select_returns_json() {
        let mut client = default_client();
        let q = select_query("SELECT ?s WHERE { ?s ?p ?o }");
        let resp = client.execute(q).expect("execute failed");
        assert!(resp.result_json.contains("results"));
        assert!(resp.result_json.contains("bindings"));
    }

    #[test]
    fn test_execute_select_contains_variables() {
        let mut client = default_client();
        let q = select_query("SELECT ?name ?age WHERE { ?x ?name ?age }");
        let resp = client.execute(q).expect("execute failed");
        assert!(resp.result_json.contains("name"), "json: {}", resp.result_json);
        assert!(resp.result_json.contains("age"), "json: {}", resp.result_json);
    }

    #[test]
    fn test_execute_select_query_type_preserved() {
        let mut client = default_client();
        let q = select_query("SELECT ?s WHERE { ?s ?p ?o }");
        let resp = client.execute(q).expect("execute");
        assert_eq!(resp.query_type, QueryType::Select);
    }

    // ── execute ASK ────────────────────────────────────────────────────────

    #[test]
    fn test_execute_ask_returns_boolean() {
        let mut client = default_client();
        let q = SparqlQuery {
            text: "ASK { ?s ?p ?o }".to_string(),
            query_type: QueryType::Ask,
            timeout_ms: None,
            default_graph: None,
        };
        let resp = client.execute(q).expect("execute");
        assert!(resp.result_json.contains("boolean"));
        assert_eq!(resp.query_type, QueryType::Ask);
    }

    #[test]
    fn test_execute_ask_response_has_head() {
        let mut client = default_client();
        let q = SparqlQuery {
            text: "ASK { <x> <p> <o> }".to_string(),
            query_type: QueryType::Ask,
            timeout_ms: None,
            default_graph: None,
        };
        let resp = client.execute(q).expect("execute");
        assert!(resp.result_json.contains("head"));
    }

    // ── execute CONSTRUCT ──────────────────────────────────────────────────

    #[test]
    fn test_execute_construct_returns_triples() {
        let mut client = default_client();
        let q = SparqlQuery {
            text: "CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }".to_string(),
            query_type: QueryType::Construct,
            timeout_ms: None,
            default_graph: None,
        };
        let resp = client.execute(q).expect("execute");
        assert!(resp.result_json.contains("triples") || resp.result_json.contains("results"));
        assert_eq!(resp.query_type, QueryType::Construct);
    }

    #[test]
    fn test_execute_describe_returns_graph() {
        let mut client = default_client();
        let q = SparqlQuery {
            text: "DESCRIBE <http://example.org/x>".to_string(),
            query_type: QueryType::Describe,
            timeout_ms: None,
            default_graph: None,
        };
        let resp = client.execute(q).expect("execute");
        assert!(!resp.result_json.is_empty());
    }

    // ── detect_query_type ──────────────────────────────────────────────────

    #[test]
    fn test_detect_select() {
        assert_eq!(
            EndpointClient::detect_query_type("SELECT ?s WHERE { ?s ?p ?o }"),
            QueryType::Select
        );
    }

    #[test]
    fn test_detect_ask() {
        assert_eq!(
            EndpointClient::detect_query_type("ASK { ?s ?p ?o }"),
            QueryType::Ask
        );
    }

    #[test]
    fn test_detect_construct() {
        assert_eq!(
            EndpointClient::detect_query_type("CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }"),
            QueryType::Construct
        );
    }

    #[test]
    fn test_detect_describe() {
        assert_eq!(
            EndpointClient::detect_query_type("DESCRIBE <http://example.org/x>"),
            QueryType::Describe
        );
    }

    #[test]
    fn test_detect_case_insensitive() {
        assert_eq!(
            EndpointClient::detect_query_type("ask { ?s ?p ?o }"),
            QueryType::Ask
        );
    }

    #[test]
    fn test_detect_default_is_select() {
        // No recognised keyword → SELECT
        assert_eq!(
            EndpointClient::detect_query_type("FROM <g> WHERE { ?s ?p ?o }"),
            QueryType::Select
        );
    }

    // ── stats ──────────────────────────────────────────────────────────────

    #[test]
    fn test_stats_initial_zero() {
        let client = default_client();
        assert_eq!(client.stats().total_requests, 0);
        assert_eq!(client.stats().successful, 0);
        assert_eq!(client.stats().failed, 0);
    }

    #[test]
    fn test_stats_increment_on_success() {
        let mut client = default_client();
        client.execute(select_query("SELECT ?s WHERE { ?s ?p ?o }")).expect("should succeed");
        assert_eq!(client.stats().total_requests, 1);
        assert_eq!(client.stats().successful, 1);
        assert_eq!(client.stats().failed, 0);
    }

    #[test]
    fn test_stats_increment_failed_on_invalid() {
        let mut client = default_client();
        let _ = client.execute(select_query(""));
        assert_eq!(client.stats().failed, 1);
        assert_eq!(client.stats().successful, 0);
    }

    #[test]
    fn test_stats_total_elapsed_grows() {
        let mut client = default_client();
        client.execute(select_query("SELECT ?x WHERE { ?x ?p ?o }")).expect("should succeed");
        assert!(client.stats().total_elapsed_ms > 0);
    }

    #[test]
    fn test_stats_multiple_queries() {
        let mut client = default_client();
        for _ in 0..5 {
            client.execute(select_query("SELECT ?s WHERE { ?s ?p ?o }")).expect("should succeed");
        }
        assert_eq!(client.stats().total_requests, 5);
        assert_eq!(client.stats().successful, 5);
    }

    // ── history ────────────────────────────────────────────────────────────

    #[test]
    fn test_history_grows_on_execute() {
        let mut client = default_client();
        assert_eq!(client.history().len(), 0);
        client.execute(select_query("SELECT ?s WHERE { ?s ?p ?o }")).expect("should succeed");
        assert_eq!(client.history().len(), 1);
    }

    #[test]
    fn test_clear_history() {
        let mut client = default_client();
        client.execute(select_query("SELECT ?s WHERE { ?s ?p ?o }")).expect("should succeed");
        client.clear_history();
        assert_eq!(client.history().len(), 0);
    }

    #[test]
    fn test_last_query_none_initially() {
        let client = default_client();
        assert!(client.last_query().is_none());
    }

    #[test]
    fn test_last_query_after_execute() {
        let mut client = default_client();
        client.execute(select_query("SELECT ?x WHERE { ?x ?p ?o }")).expect("should succeed");
        let last = client.last_query().expect("last query");
        assert!(last.text.contains("?x"));
    }

    #[test]
    fn test_history_records_query_text() {
        let mut client = default_client();
        let text = "SELECT ?name WHERE { ?s :name ?name }";
        client.execute(select_query(text)).expect("should succeed");
        assert_eq!(client.history()[0].text, text);
    }

    // ── execute_update ─────────────────────────────────────────────────────

    #[test]
    fn test_execute_update_success() {
        let mut client = default_client();
        let result = client.execute_update("INSERT DATA { <a> <b> <c> }");
        assert!(result.is_ok());
    }

    #[test]
    fn test_execute_update_increments_stats() {
        let mut client = default_client();
        client.execute_update("DELETE DATA { <a> <b> <c> }").expect("should succeed");
        assert_eq!(client.stats().total_requests, 1);
        assert_eq!(client.stats().successful, 1);
    }

    #[test]
    fn test_execute_update_empty_fails() {
        let mut client = default_client();
        assert!(client.execute_update("").is_err());
    }

    // ── build_select_response ──────────────────────────────────────────────

    #[test]
    fn test_build_select_response_contains_vars() {
        let json = EndpointClient::build_select_response(&["s", "p", "o"], 2);
        assert!(json.contains(r#""s""#));
        assert!(json.contains(r#""p""#));
        assert!(json.contains(r#""o""#));
    }

    #[test]
    fn test_build_select_response_row_count() {
        let json = EndpointClient::build_select_response(&["x"], 3);
        // Three binding objects: count occurrences of "value_x_"
        let count = json.matches("value_x_").count();
        assert_eq!(count, 3, "json = {}", json);
    }

    #[test]
    fn test_build_select_response_empty_vars() {
        let json = EndpointClient::build_select_response(&[], 0);
        assert!(json.contains("results"));
    }

    // ── timeout ────────────────────────────────────────────────────────────

    #[test]
    fn test_execute_timeout_marker() {
        let mut client = default_client();
        let q = select_query("SELECT __timeout__ ?s WHERE { ?s ?p ?o }");
        let err = client.execute(q).expect_err("should time out");
        assert!(matches!(err, ClientError::Timeout));
    }

    // ── config ─────────────────────────────────────────────────────────────

    #[test]
    fn test_client_config_default() {
        let cfg = ClientConfig::default();
        assert!(!cfg.endpoint_url.is_empty());
        assert!(cfg.default_timeout_ms > 0);
    }

    #[test]
    fn test_client_custom_config() {
        let cfg = ClientConfig {
            endpoint_url: "http://custom:8080/sparql".to_string(),
            default_timeout_ms: 1000,
            max_retries: 1,
            accept_format: "text/csv".to_string(),
        };
        let client = EndpointClient::new(cfg.clone());
        assert_eq!(client.config.endpoint_url, "http://custom:8080/sparql");
    }

    // ── Additional coverage ─────────────────────────────────────────────────

    #[test]
    fn test_execute_row_count_positive() {
        let mut client = default_client();
        let q = select_query("SELECT ?s ?p ?o WHERE { ?s ?p ?o }");
        let resp = client.execute(q).expect("execute");
        assert!(resp.row_count > 0);
    }

    #[test]
    fn test_execute_elapsed_ms_positive() {
        let mut client = default_client();
        let q = select_query("SELECT ?s WHERE { ?s ?p ?o }");
        let resp = client.execute(q).expect("execute");
        assert!(resp.elapsed_ms > 0);
    }

    #[test]
    fn test_failed_query_not_in_history() {
        let mut client = default_client();
        let _ = client.execute(select_query(""));
        // Empty query should fail and NOT be added to history
        assert_eq!(client.history().len(), 0);
    }

    #[test]
    fn test_history_preserves_query_type() {
        let mut client = default_client();
        let q = SparqlQuery {
            text: "ASK { ?s ?p ?o }".to_string(),
            query_type: QueryType::Ask,
            timeout_ms: None,
            default_graph: None,
        };
        client.execute(q).expect("execute");
        assert_eq!(client.history()[0].query_type, QueryType::Ask);
    }

    #[test]
    fn test_clear_history_after_multiple() {
        let mut client = default_client();
        for _ in 0..3 {
            client.execute(select_query("SELECT ?s WHERE { ?s ?p ?o }")).expect("should succeed");
        }
        assert_eq!(client.history().len(), 3);
        client.clear_history();
        assert_eq!(client.history().len(), 0);
    }

    #[test]
    fn test_build_select_response_head_vars() {
        let json = EndpointClient::build_select_response(&["name", "age"], 1);
        assert!(json.contains("head"), "json = {}", json);
        assert!(json.contains("vars"), "json = {}", json);
    }

    #[test]
    fn test_detect_construct_lowercase() {
        assert_eq!(
            EndpointClient::detect_query_type("construct { ?s ?p ?o } where { ?s ?p ?o }"),
            QueryType::Construct
        );
    }

    #[test]
    fn test_detect_describe_lowercase() {
        assert_eq!(
            EndpointClient::detect_query_type("describe <http://example.org/x>"),
            QueryType::Describe
        );
    }

    #[test]
    fn test_stats_failed_on_timeout() {
        let mut client = default_client();
        let _ = client.execute(select_query("SELECT __timeout__ WHERE { }"));
        assert_eq!(client.stats().failed, 1);
        assert_eq!(client.stats().successful, 0);
    }

    #[test]
    fn test_execute_with_default_graph() {
        let mut client = default_client();
        let q = SparqlQuery {
            text: "SELECT ?s WHERE { ?s ?p ?o }".to_string(),
            query_type: QueryType::Select,
            timeout_ms: Some(5000),
            default_graph: Some("http://example.org/graph".to_string()),
        };
        let resp = client.execute(q).expect("execute");
        assert_eq!(resp.query_type, QueryType::Select);
    }

    #[test]
    fn test_request_stats_total_is_success_plus_failed() {
        let mut client = default_client();
        client.execute(select_query("SELECT ?s WHERE { ?s ?p ?o }")).expect("should succeed");
        let _ = client.execute(select_query(""));
        let s = client.stats();
        assert_eq!(s.total_requests, s.successful + s.failed);
    }

    #[test]
    fn test_client_error_display_network() {
        let err = ClientError::NetworkError("connection refused".to_string());
        let msg = err.to_string();
        assert!(msg.contains("connection refused"));
    }

    #[test]
    fn test_client_error_display_timeout() {
        let err = ClientError::Timeout;
        let msg = err.to_string();
        assert!(msg.contains("timed out"));
    }

    #[test]
    fn test_client_error_display_parse() {
        let err = ClientError::ParseError("unexpected token".to_string());
        let msg = err.to_string();
        assert!(msg.contains("unexpected token"));
    }

    #[test]
    fn test_client_error_display_invalid_query() {
        let err = ClientError::InvalidQuery("syntax error".to_string());
        let msg = err.to_string();
        assert!(msg.contains("syntax error"));
    }
}
