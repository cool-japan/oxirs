//! RDF graph updater for persisting generated triples
//!
//! Provides SPARQL UPDATE execution with batch operations,
//! named graph support, and automatic retry logic.

use super::triple_generator::GeneratedTriple;
#[cfg(feature = "http-client")]
use crate::error::ModbusError;
use crate::error::ModbusResult;
use oxirs_core::model::Triple;
use std::time::Duration;

/// SPARQL endpoint configuration
#[derive(Debug, Clone)]
pub struct SparqlEndpointConfig {
    /// SPARQL UPDATE endpoint URL (e.g., "http://localhost:3030/dataset/update")
    pub update_url: String,
    /// Optional SPARQL query endpoint for verification
    pub query_url: Option<String>,
    /// Connection timeout
    pub connect_timeout: Duration,
    /// Request timeout
    pub request_timeout: Duration,
    /// Maximum retry attempts on failure
    pub max_retries: u32,
    /// Delay between retry attempts
    pub retry_delay: Duration,
    /// Optional HTTP Basic Auth username
    pub username: Option<String>,
    /// Optional HTTP Basic Auth password
    pub password: Option<String>,
}

impl Default for SparqlEndpointConfig {
    fn default() -> Self {
        Self {
            update_url: "http://localhost:3030/dataset/update".to_string(),
            query_url: None,
            connect_timeout: Duration::from_secs(5),
            request_timeout: Duration::from_secs(30),
            max_retries: 3,
            retry_delay: Duration::from_secs(1),
            username: None,
            password: None,
        }
    }
}

impl SparqlEndpointConfig {
    /// Create a new config with just the update URL
    pub fn new(update_url: impl Into<String>) -> Self {
        Self {
            update_url: update_url.into(),
            ..Default::default()
        }
    }

    /// Set authentication credentials
    pub fn with_auth(mut self, username: impl Into<String>, password: impl Into<String>) -> Self {
        self.username = Some(username.into());
        self.password = Some(password.into());
        self
    }

    /// Set the named graph for updates
    pub fn with_query_url(mut self, query_url: impl Into<String>) -> Self {
        self.query_url = Some(query_url.into());
        self
    }
}

/// Statistics for graph update operations
#[derive(Debug, Clone, Default)]
pub struct UpdateStats {
    /// Total updates attempted
    pub total_updates: u64,
    /// Successful updates
    pub successful_updates: u64,
    /// Failed updates
    pub failed_updates: u64,
    /// Total triples inserted
    pub triples_inserted: u64,
    /// Total retries performed
    pub retries: u64,
    /// Last error message if any
    pub last_error: Option<String>,
}

/// RDF graph updater for SPARQL endpoints
///
/// Manages batch insertion of triples with automatic retry logic.
pub struct GraphUpdater {
    /// Endpoint configuration (used by http-client feature)
    #[allow(dead_code)]
    config: SparqlEndpointConfig,
    /// Named graph IRI (optional)
    graph_iri: Option<String>,
    /// Batch size for INSERT DATA
    batch_size: usize,
    /// Update statistics
    stats: UpdateStats,
    /// HTTP client (lazy initialized)
    #[cfg(feature = "http-client")]
    client: Option<reqwest::Client>,
}

impl GraphUpdater {
    /// Create a new graph updater
    pub fn new(config: SparqlEndpointConfig) -> Self {
        Self {
            config,
            graph_iri: None,
            batch_size: 100,
            stats: UpdateStats::default(),
            #[cfg(feature = "http-client")]
            client: None,
        }
    }

    /// Set the target named graph
    pub fn with_graph(mut self, graph_iri: impl Into<String>) -> Self {
        self.graph_iri = Some(graph_iri.into());
        self
    }

    /// Set the batch size for INSERT DATA operations
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size.max(1);
        self
    }

    /// Get update statistics
    pub fn stats(&self) -> &UpdateStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = UpdateStats::default();
    }

    /// Build SPARQL INSERT DATA query for a batch of triples
    pub fn build_insert_query(&self, triples: &[Triple]) -> String {
        let mut query = String::new();

        if let Some(ref graph_iri) = self.graph_iri {
            query.push_str(&format!("INSERT DATA {{ GRAPH <{}> {{\n", graph_iri));
        } else {
            query.push_str("INSERT DATA {\n");
        }

        for triple in triples {
            query.push_str("  ");
            query.push_str(&triple_to_turtle(triple));
            query.push_str(" .\n");
        }

        if self.graph_iri.is_some() {
            query.push_str("} }");
        } else {
            query.push('}');
        }

        query
    }

    /// Build SPARQL INSERT DATA query including provenance triples
    pub fn build_insert_with_provenance(&self, generated: &[GeneratedTriple]) -> String {
        let mut all_triples: Vec<Triple> = Vec::new();

        for gen in generated {
            all_triples.push(gen.triple.clone());
            all_triples.extend(gen.provenance_triples());
        }

        self.build_insert_query(&all_triples)
    }

    /// Insert triples using SPARQL UPDATE (local execution without HTTP)
    ///
    /// This version builds the query string but doesn't execute it.
    /// Use `insert_triples_http` for actual HTTP execution.
    pub fn insert_triples_local(&mut self, triples: &[Triple]) -> ModbusResult<String> {
        if triples.is_empty() {
            return Ok(String::new());
        }

        self.stats.total_updates += 1;

        let query = self.build_insert_query(triples);
        self.stats.successful_updates += 1;
        self.stats.triples_inserted += triples.len() as u64;

        Ok(query)
    }

    /// Insert generated triples with provenance (local execution)
    pub fn insert_generated_local(
        &mut self,
        generated: &[GeneratedTriple],
    ) -> ModbusResult<String> {
        if generated.is_empty() {
            return Ok(String::new());
        }

        self.stats.total_updates += 1;

        let query = self.build_insert_with_provenance(generated);

        // Count all triples including provenance
        let total_triples: usize = generated
            .iter()
            .map(|g| 1 + g.provenance_triples().len())
            .sum();

        self.stats.successful_updates += 1;
        self.stats.triples_inserted += total_triples as u64;

        Ok(query)
    }

    /// Insert triples in batches
    pub fn insert_batches(&mut self, triples: &[Triple]) -> ModbusResult<Vec<String>> {
        let mut queries = Vec::new();

        for chunk in triples.chunks(self.batch_size) {
            let query = self.insert_triples_local(chunk)?;
            if !query.is_empty() {
                queries.push(query);
            }
        }

        Ok(queries)
    }

    /// Execute SPARQL UPDATE via HTTP
    #[cfg(feature = "http-client")]
    pub async fn insert_triples_http(&mut self, triples: &[Triple]) -> ModbusResult<()> {
        use tokio::time::sleep;

        if triples.is_empty() {
            return Ok(());
        }

        // Initialize client if needed
        if self.client.is_none() {
            let client = reqwest::Client::builder()
                .connect_timeout(self.config.connect_timeout)
                .timeout(self.config.request_timeout)
                .build()
                .map_err(|e| {
                    ModbusError::Io(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        format!("Failed to create HTTP client: {}", e),
                    ))
                })?;
            self.client = Some(client);
        }

        let query = self.build_insert_query(triples);
        self.stats.total_updates += 1;

        let mut last_error = None;

        for attempt in 0..=self.config.max_retries {
            if attempt > 0 {
                self.stats.retries += 1;
                sleep(self.config.retry_delay).await;
            }

            let mut request = self
                .client
                .as_ref()
                .unwrap()
                .post(&self.config.update_url)
                .header("Content-Type", "application/sparql-update")
                .body(query.clone());

            // Add auth if configured
            if let (Some(ref user), Some(ref pass)) = (&self.config.username, &self.config.password)
            {
                request = request.basic_auth(user, Some(pass));
            }

            match request.send().await {
                Ok(response) => {
                    if response.status().is_success() {
                        self.stats.successful_updates += 1;
                        self.stats.triples_inserted += triples.len() as u64;
                        return Ok(());
                    } else {
                        let status = response.status();
                        let body = response
                            .text()
                            .await
                            .unwrap_or_else(|_| "Unknown error".to_string());
                        last_error = Some(format!("HTTP {}: {}", status, body));
                    }
                }
                Err(e) => {
                    last_error = Some(e.to_string());
                }
            }
        }

        self.stats.failed_updates += 1;
        let error_msg = last_error.unwrap_or_else(|| "Unknown error".to_string());
        self.stats.last_error = Some(error_msg.clone());

        Err(ModbusError::Io(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("SPARQL UPDATE failed: {}", error_msg),
        )))
    }

    /// Execute HTTP update for generated triples with provenance
    #[cfg(feature = "http-client")]
    pub async fn insert_generated_http(
        &mut self,
        generated: &[GeneratedTriple],
    ) -> ModbusResult<()> {
        let mut all_triples: Vec<Triple> = Vec::new();

        for gen in generated {
            all_triples.push(gen.triple.clone());
            all_triples.extend(gen.provenance_triples());
        }

        self.insert_triples_http(&all_triples).await
    }
}

/// Convert a triple to Turtle N-Triples format
fn triple_to_turtle(triple: &Triple) -> String {
    use oxirs_core::model::{Object, Predicate, Subject};

    let subject = match triple.subject() {
        Subject::NamedNode(n) => format!("<{}>", n.as_str()),
        Subject::BlankNode(b) => format!("_:{}", b.as_str()),
        Subject::Variable(v) => format!("?{}", v.as_str()),
        Subject::QuotedTriple(qt) => format!("<< {} >>", triple_to_turtle(qt.inner())),
    };

    let predicate = match triple.predicate() {
        Predicate::NamedNode(n) => format!("<{}>", n.as_str()),
        Predicate::Variable(v) => format!("?{}", v.as_str()),
    };

    let object = match triple.object() {
        Object::NamedNode(n) => format!("<{}>", n.as_str()),
        Object::BlankNode(b) => format!("_:{}", b.as_str()),
        Object::Literal(lit) => {
            // Format literal with proper escaping
            let value = lit.value().replace('\\', "\\\\").replace('"', "\\\"");
            if let Some(lang) = lit.language() {
                format!("\"{}\"@{}", value, lang)
            } else {
                // All literals have a datatype
                let dt = lit.datatype();
                format!("\"{}\"^^<{}>", value, dt.as_str())
            }
        }
        Object::Variable(v) => format!("?{}", v.as_str()),
        Object::QuotedTriple(qt) => {
            // RDF-star quoted triple
            format!("<< {} >>", triple_to_turtle(qt.inner()))
        }
    };

    format!("{} {} {}", subject, predicate, object)
}

/// Builder for batch update operations
pub struct BatchUpdater {
    /// Graph updater
    updater: GraphUpdater,
    /// Pending triples to insert
    pending: Vec<Triple>,
    /// Auto-flush threshold
    flush_threshold: usize,
}

impl BatchUpdater {
    /// Create a new batch updater
    pub fn new(config: SparqlEndpointConfig) -> Self {
        Self {
            updater: GraphUpdater::new(config),
            pending: Vec::new(),
            flush_threshold: 100,
        }
    }

    /// Set the named graph
    pub fn with_graph(mut self, graph_iri: impl Into<String>) -> Self {
        self.updater = self.updater.with_graph(graph_iri);
        self
    }

    /// Set auto-flush threshold
    pub fn with_flush_threshold(mut self, threshold: usize) -> Self {
        self.flush_threshold = threshold.max(1);
        self
    }

    /// Add a triple to the pending batch
    pub fn add(&mut self, triple: Triple) {
        self.pending.push(triple);
    }

    /// Add generated triples with provenance
    pub fn add_generated(&mut self, generated: &GeneratedTriple) {
        self.pending.push(generated.triple.clone());
        self.pending.extend(generated.provenance_triples());
    }

    /// Check if auto-flush is needed
    pub fn should_flush(&self) -> bool {
        self.pending.len() >= self.flush_threshold
    }

    /// Get pending count
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Flush pending triples (local - returns query string)
    pub fn flush_local(&mut self) -> ModbusResult<Option<String>> {
        if self.pending.is_empty() {
            return Ok(None);
        }

        let triples = std::mem::take(&mut self.pending);
        let query = self.updater.insert_triples_local(&triples)?;

        if query.is_empty() {
            Ok(None)
        } else {
            Ok(Some(query))
        }
    }

    /// Flush pending triples via HTTP
    #[cfg(feature = "http-client")]
    pub async fn flush_http(&mut self) -> ModbusResult<()> {
        if self.pending.is_empty() {
            return Ok(());
        }

        let triples = std::mem::take(&mut self.pending);
        self.updater.insert_triples_http(&triples).await
    }

    /// Get statistics
    pub fn stats(&self) -> &UpdateStats {
        self.updater.stats()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxirs_core::model::{Literal, NamedNode};

    fn create_test_triple() -> Triple {
        let subject = NamedNode::new("http://example.com/device/plc001").unwrap();
        let predicate = NamedNode::new("http://example.com/property/temperature").unwrap();
        let datatype = NamedNode::new("http://www.w3.org/2001/XMLSchema#float").unwrap();
        let literal = Literal::new_typed("22.5", datatype);
        Triple::new(subject, predicate, literal)
    }

    #[test]
    fn test_build_insert_query() {
        let config = SparqlEndpointConfig::default();
        let updater = GraphUpdater::new(config);

        let triple = create_test_triple();
        let query = updater.build_insert_query(&[triple]);

        assert!(query.contains("INSERT DATA"));
        assert!(query.contains("http://example.com/device/plc001"));
        assert!(query.contains("http://example.com/property/temperature"));
        assert!(query.contains("22.5"));
    }

    #[test]
    fn test_build_insert_with_graph() {
        let config = SparqlEndpointConfig::default();
        let updater = GraphUpdater::new(config).with_graph("http://example.com/graph/modbus");

        let triple = create_test_triple();
        let query = updater.build_insert_query(&[triple]);

        assert!(query.contains("GRAPH <http://example.com/graph/modbus>"));
    }

    #[test]
    fn test_insert_triples_local() {
        let config = SparqlEndpointConfig::default();
        let mut updater = GraphUpdater::new(config);

        let triple = create_test_triple();
        let query = updater.insert_triples_local(&[triple]).unwrap();

        assert!(!query.is_empty());
        assert_eq!(updater.stats().successful_updates, 1);
        assert_eq!(updater.stats().triples_inserted, 1);
    }

    #[test]
    fn test_insert_batches() {
        let config = SparqlEndpointConfig::default();
        let mut updater = GraphUpdater::new(config).with_batch_size(2);

        let triples: Vec<Triple> = (0..5).map(|_| create_test_triple()).collect();
        let queries = updater.insert_batches(&triples).unwrap();

        // 5 triples / 2 batch size = 3 batches
        assert_eq!(queries.len(), 3);
        assert_eq!(updater.stats().successful_updates, 3);
        assert_eq!(updater.stats().triples_inserted, 5);
    }

    #[test]
    fn test_batch_updater() {
        let config = SparqlEndpointConfig::default();
        let mut batch = BatchUpdater::new(config).with_flush_threshold(3);

        batch.add(create_test_triple());
        batch.add(create_test_triple());
        assert!(!batch.should_flush());

        batch.add(create_test_triple());
        assert!(batch.should_flush());

        let query = batch.flush_local().unwrap();
        assert!(query.is_some());
        assert_eq!(batch.pending_count(), 0);
    }

    #[test]
    fn test_triple_to_turtle() {
        let triple = create_test_triple();
        let turtle = triple_to_turtle(&triple);

        assert!(turtle.starts_with('<'));
        assert!(turtle.contains("http://example.com/device/plc001"));
        assert!(turtle.contains("^^<http://www.w3.org/2001/XMLSchema#float>"));
    }

    #[test]
    fn test_endpoint_config() {
        let config = SparqlEndpointConfig::new("http://localhost:3030/update")
            .with_auth("admin", "secret123")
            .with_query_url("http://localhost:3030/query");

        assert_eq!(config.update_url, "http://localhost:3030/update");
        assert_eq!(config.username.as_deref(), Some("admin"));
        assert_eq!(config.password.as_deref(), Some("secret123"));
        assert_eq!(
            config.query_url.as_deref(),
            Some("http://localhost:3030/query")
        );
    }

    #[test]
    fn test_empty_insert() {
        let config = SparqlEndpointConfig::default();
        let mut updater = GraphUpdater::new(config);

        let query = updater.insert_triples_local(&[]).unwrap();
        assert!(query.is_empty());
        assert_eq!(updater.stats().total_updates, 0);
    }

    #[test]
    fn test_stats_reset() {
        let config = SparqlEndpointConfig::default();
        let mut updater = GraphUpdater::new(config);

        let triple = create_test_triple();
        updater.insert_triples_local(&[triple]).unwrap();
        assert_eq!(updater.stats().successful_updates, 1);

        updater.reset_stats();
        assert_eq!(updater.stats().successful_updates, 0);
    }
}
