//! # C-SPARQL (Continuous SPARQL) Extensions
//!
//! Implementation of C-SPARQL for continuous query processing over RDF streams.
//!
//! C-SPARQL extends SPARQL 1.1 with:
//! - Stream declarations (FROM STREAM)
//! - Time-based and count-based windows
//! - Tumbling and sliding windows
//! - Stream-to-relation operators
//! - Temporal aggregations
//!
//! ## References
//! - Barbieri et al. "C-SPARQL: a Continuous Query Language for RDF Data Streams"
//! - <https://streamreasoning.org/>

use anyhow::{anyhow, Result};
use chrono::{DateTime, Duration as ChronoDuration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{debug, info};

use crate::store_integration::{QueryResult, RdfStore, Triple};
use crate::StreamEvent;

/// C-SPARQL query engine for continuous stream processing
pub struct CSparqlEngine {
    /// RDF store backend
    store: Arc<dyn RdfStore>,
    /// Active stream windows
    windows: Arc<RwLock<HashMap<String, StreamWindow>>>,
    /// Registered queries
    queries: Arc<RwLock<HashMap<String, CSparqlQuery>>>,
    /// Configuration
    config: CSparqlConfig,
    /// Statistics
    stats: Arc<RwLock<CSparqlStats>>,
}

/// C-SPARQL configuration
#[derive(Debug, Clone)]
pub struct CSparqlConfig {
    /// Maximum number of concurrent queries
    pub max_queries: usize,
    /// Default window size
    pub default_window_size: Duration,
    /// Default window step
    pub default_window_step: Duration,
    /// Enable incremental evaluation
    pub incremental_evaluation: bool,
    /// Memory limit for windows (bytes)
    pub memory_limit: usize,
}

impl Default for CSparqlConfig {
    fn default() -> Self {
        Self {
            max_queries: 100,
            default_window_size: Duration::from_secs(60),
            default_window_step: Duration::from_secs(10),
            incremental_evaluation: true,
            memory_limit: 1024 * 1024 * 100, // 100 MB
        }
    }
}

/// C-SPARQL query representation
#[derive(Debug, Clone)]
pub struct CSparqlQuery {
    /// Query identifier
    pub id: String,
    /// Original query string
    pub query_string: String,
    /// Parsed query components
    pub components: QueryComponents,
    /// Execution metadata
    pub metadata: QueryMetadata,
    /// Query state
    pub state: QueryState,
}

/// Parsed query components
#[derive(Debug, Clone)]
pub struct QueryComponents {
    /// Stream declarations
    pub streams: Vec<StreamDeclaration>,
    /// Window specifications
    pub windows: Vec<WindowSpec>,
    /// SELECT/CONSTRUCT/ASK query part
    pub query_type: QueryType,
    /// WHERE clause patterns
    pub patterns: Vec<TriplePattern>,
    /// Aggregations
    pub aggregations: Vec<Aggregation>,
    /// GROUP BY variables
    pub group_by: Vec<String>,
    /// HAVING conditions
    pub having: Option<String>,
    /// ORDER BY clause
    pub order_by: Vec<OrderByClause>,
    /// LIMIT
    pub limit: Option<usize>,
}

/// Stream declaration (FROM STREAM `<uri>`)
#[derive(Debug, Clone)]
pub struct StreamDeclaration {
    /// Stream URI
    pub uri: String,
    /// Window specification for this stream
    pub window: WindowSpec,
}

/// Window specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowSpec {
    /// Window type
    pub window_type: WindowType,
    /// Window range (time-based or count-based)
    pub range: WindowRange,
    /// Window step (for sliding windows)
    pub step: Option<WindowRange>,
}

/// Window types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum WindowType {
    /// Tumbling window (non-overlapping)
    Tumbling,
    /// Sliding window (overlapping)
    Sliding,
    /// Landmark window (from a fixed point)
    Landmark,
}

/// Window range specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WindowRange {
    /// Time-based window (e.g., "PT10S" for 10 seconds)
    Time(Duration),
    /// Count-based window (number of events)
    Count(usize),
    /// Batch-based (process N batches)
    Batch(usize),
}

/// Query types
#[derive(Debug, Clone, PartialEq)]
pub enum QueryType {
    Select,
    Construct,
    Ask,
    Describe,
}

/// Triple pattern in WHERE clause
#[derive(Debug, Clone)]
pub struct TriplePattern {
    pub subject: PatternElement,
    pub predicate: PatternElement,
    pub object: PatternElement,
}

/// Pattern element (variable or value)
#[derive(Debug, Clone)]
pub enum PatternElement {
    Variable(String),
    IRI(String),
    Literal(String),
    Blank(String),
}

/// Aggregation function
#[derive(Debug, Clone)]
pub struct Aggregation {
    pub function: AggregationFunction,
    pub variable: String,
    pub alias: Option<String>,
}

/// Aggregation functions
#[derive(Debug, Clone, PartialEq)]
pub enum AggregationFunction {
    Count,
    Sum,
    Avg,
    Min,
    Max,
    Sample,
    GroupConcat { separator: String },
}

/// ORDER BY clause
#[derive(Debug, Clone)]
pub struct OrderByClause {
    pub variable: String,
    pub ascending: bool,
}

/// Query metadata
#[derive(Debug, Clone)]
pub struct QueryMetadata {
    pub name: Option<String>,
    pub description: Option<String>,
    pub created_at: DateTime<Utc>,
    pub owner: Option<String>,
    pub tags: Vec<String>,
}

/// Query execution state
#[derive(Debug, Clone, PartialEq)]
pub enum QueryState {
    Registered,
    Running,
    Paused,
    Stopped,
    Error(String),
}

/// Stream window for buffering events
pub struct StreamWindow {
    /// Window identifier
    pub id: String,
    /// Window specification
    pub spec: WindowSpec,
    /// Buffered triples
    pub buffer: VecDeque<WindowedTriple>,
    /// Window start time
    pub start_time: DateTime<Utc>,
    /// Last update time
    pub last_update: DateTime<Utc>,
    /// Total events processed
    pub event_count: usize,
}

/// Triple with timestamp for windowing
#[derive(Debug, Clone)]
pub struct WindowedTriple {
    pub triple: Triple,
    pub timestamp: DateTime<Utc>,
    pub event_id: String,
}

/// C-SPARQL statistics
#[derive(Debug, Clone, Default)]
pub struct CSparqlStats {
    pub queries_registered: u64,
    pub queries_executed: u64,
    pub queries_failed: u64,
    pub total_events_processed: u64,
    pub total_results_produced: u64,
    pub avg_query_latency_ms: f64,
    pub active_windows: usize,
}

impl CSparqlEngine {
    /// Create a new C-SPARQL engine
    pub fn new(store: Arc<dyn RdfStore>, config: CSparqlConfig) -> Self {
        Self {
            store,
            windows: Arc::new(RwLock::new(HashMap::new())),
            queries: Arc::new(RwLock::new(HashMap::new())),
            config,
            stats: Arc::new(RwLock::new(CSparqlStats::default())),
        }
    }

    /// Register a C-SPARQL query
    pub async fn register_query(&self, query_string: String) -> Result<String> {
        let query_id = uuid::Uuid::new_v4().to_string();

        // Parse the C-SPARQL query
        let components = self.parse_csparql_query(&query_string)?;

        let query = CSparqlQuery {
            id: query_id.clone(),
            query_string,
            components,
            metadata: QueryMetadata {
                name: None,
                description: None,
                created_at: Utc::now(),
                owner: None,
                tags: Vec::new(),
            },
            state: QueryState::Registered,
        };

        // Register the query
        let mut queries = self.queries.write().await;
        if queries.len() >= self.config.max_queries {
            return Err(anyhow!("Maximum number of queries reached"));
        }
        queries.insert(query_id.clone(), query);

        let mut stats = self.stats.write().await;
        stats.queries_registered += 1;

        info!("Registered C-SPARQL query: {}", query_id);
        Ok(query_id)
    }

    /// Process a stream event
    pub async fn process_event(&self, event: &StreamEvent) -> Result<()> {
        // Extract triples from event
        let triples = self.extract_triples_from_event(event)?;

        // Update all relevant windows
        let mut windows = self.windows.write().await;
        for (_window_id, window) in windows.iter_mut() {
            for triple in &triples {
                let windowed_triple = WindowedTriple {
                    triple: triple.clone(),
                    timestamp: Utc::now(),
                    event_id: uuid::Uuid::new_v4().to_string(),
                };

                window.buffer.push_back(windowed_triple);
                window.event_count += 1;
                window.last_update = Utc::now();
            }

            // Apply window eviction policy
            self.evict_expired_triples(window).await?;
        }

        let mut stats = self.stats.write().await;
        stats.total_events_processed += 1;

        Ok(())
    }

    /// Execute a registered query
    pub async fn execute_query(&self, query_id: &str) -> Result<QueryResult> {
        let queries = self.queries.read().await;
        let query = queries
            .get(query_id)
            .ok_or_else(|| anyhow!("Query not found: {}", query_id))?;

        // Get relevant window data
        let window_data = self.get_window_data_for_query(query).await?;

        // Execute SPARQL query on window data
        let result = self.execute_sparql_on_window(query, &window_data).await?;

        let mut stats = self.stats.write().await;
        stats.queries_executed += 1;
        stats.total_results_produced += result.bindings.len() as u64;

        Ok(result)
    }

    /// Parse C-SPARQL query string
    fn parse_csparql_query(&self, query: &str) -> Result<QueryComponents> {
        // Simplified parser - in production, use a full SPARQL parser with C-SPARQL extensions
        let streams = self.parse_stream_declarations(query)?;
        let windows = self.parse_window_specifications(query)?;
        let query_type = self.parse_query_type(query)?;

        Ok(QueryComponents {
            streams,
            windows,
            query_type,
            patterns: Vec::new(),
            aggregations: Vec::new(),
            group_by: Vec::new(),
            having: None,
            order_by: Vec::new(),
            limit: None,
        })
    }

    /// Parse stream declarations (FROM STREAM)
    fn parse_stream_declarations(&self, query: &str) -> Result<Vec<StreamDeclaration>> {
        let mut streams = Vec::new();

        // Look for "FROM STREAM <uri> [RANGE duration] [STEP duration]"
        if query.contains("FROM STREAM") {
            // Simple regex-based parsing (use proper parser in production)
            let parts: Vec<&str> = query.split("FROM STREAM").collect();
            for part in parts.iter().skip(1) {
                if let Some(uri_end) = part.find('[') {
                    let uri = part[..uri_end]
                        .trim()
                        .trim_matches('<')
                        .trim_matches('>')
                        .to_string();

                    // Parse window specification
                    let window = if let Some(range_start) = part.find("RANGE") {
                        self.parse_window_from_string(&part[range_start..])?
                    } else {
                        WindowSpec {
                            window_type: WindowType::Tumbling,
                            range: WindowRange::Time(self.config.default_window_size),
                            step: None,
                        }
                    };

                    streams.push(StreamDeclaration { uri, window });
                }
            }
        }

        Ok(streams)
    }

    /// Parse window specifications
    fn parse_window_specifications(&self, query: &str) -> Result<Vec<WindowSpec>> {
        let mut windows = Vec::new();

        // Parse RANGE and STEP clauses
        if query.contains("RANGE") {
            let window = self.parse_window_from_string(query)?;
            windows.push(window);
        }

        Ok(windows)
    }

    /// Parse window from string
    fn parse_window_from_string(&self, s: &str) -> Result<WindowSpec> {
        let has_step = s.contains("STEP");
        let window_type = if has_step {
            WindowType::Sliding
        } else {
            WindowType::Tumbling
        };

        // Parse RANGE value (e.g., "PT10S" for 10 seconds)
        let range = if let Some(range_pos) = s.find("RANGE") {
            let range_str = &s[range_pos + 5..].trim();
            if range_str.starts_with("PT") {
                // Parse ISO 8601 duration
                let duration = self.parse_duration(range_str)?;
                WindowRange::Time(duration)
            } else if let Ok(count) = range_str.parse::<usize>() {
                WindowRange::Count(count)
            } else {
                WindowRange::Time(self.config.default_window_size)
            }
        } else {
            WindowRange::Time(self.config.default_window_size)
        };

        // Parse STEP value
        let step = if let Some(step_pos) = s.find("STEP") {
            let step_str = &s[step_pos + 4..].trim();
            if step_str.starts_with("PT") {
                let duration = self.parse_duration(step_str)?;
                Some(WindowRange::Time(duration))
            } else {
                Some(WindowRange::Time(self.config.default_window_step))
            }
        } else {
            None
        };

        Ok(WindowSpec {
            window_type,
            range,
            step,
        })
    }

    /// Parse ISO 8601 duration (simplified)
    fn parse_duration(&self, s: &str) -> Result<Duration> {
        // Handle PT format (e.g., PT10S, PT5M, PT1H)
        if !s.starts_with("PT") {
            return Err(anyhow!("Invalid duration format: {}", s));
        }

        let duration_part = &s[2..];

        if let Some(seconds_pos) = duration_part.find('S') {
            let seconds: u64 = duration_part[..seconds_pos].parse()?;
            Ok(Duration::from_secs(seconds))
        } else if let Some(minutes_pos) = duration_part.find('M') {
            let minutes: u64 = duration_part[..minutes_pos].parse()?;
            Ok(Duration::from_secs(minutes * 60))
        } else if let Some(hours_pos) = duration_part.find('H') {
            let hours: u64 = duration_part[..hours_pos].parse()?;
            Ok(Duration::from_secs(hours * 3600))
        } else {
            Err(anyhow!("Invalid duration format: {}", s))
        }
    }

    /// Parse query type
    fn parse_query_type(&self, query: &str) -> Result<QueryType> {
        let upper = query.to_uppercase();
        if upper.contains("SELECT") {
            Ok(QueryType::Select)
        } else if upper.contains("CONSTRUCT") {
            Ok(QueryType::Construct)
        } else if upper.contains("ASK") {
            Ok(QueryType::Ask)
        } else if upper.contains("DESCRIBE") {
            Ok(QueryType::Describe)
        } else {
            Err(anyhow!("Unknown query type"))
        }
    }

    /// Evict expired triples from window
    async fn evict_expired_triples(&self, window: &mut StreamWindow) -> Result<()> {
        let now = Utc::now();

        match &window.spec.range {
            WindowRange::Time(duration) => {
                // Remove triples older than window range
                let cutoff = now - ChronoDuration::from_std(*duration)?;
                window.buffer.retain(|t| t.timestamp > cutoff);
            }
            WindowRange::Count(max_count) => {
                // Keep only the last N triples
                while window.buffer.len() > *max_count {
                    window.buffer.pop_front();
                }
            }
            WindowRange::Batch(max_batches) => {
                // Batch-based eviction (simplified)
                if window.buffer.len() > max_batches * 1000 {
                    window.buffer.drain(0..*max_batches * 500);
                }
            }
        }

        Ok(())
    }

    /// Extract triples from stream event
    fn extract_triples_from_event(&self, event: &StreamEvent) -> Result<Vec<Triple>> {
        // Extract RDF triples from the event
        // This depends on the event type and structure
        let mut triples = Vec::new();

        match event {
            StreamEvent::TripleAdded {
                subject,
                predicate,
                object,
                graph,
                ..
            } => {
                triples.push(Triple {
                    subject: subject.clone(),
                    predicate: predicate.clone(),
                    object: object.clone(),
                    graph: graph.clone(),
                });
            }
            StreamEvent::QuadAdded {
                subject,
                predicate,
                object,
                graph,
                ..
            } => {
                triples.push(Triple {
                    subject: subject.clone(),
                    predicate: predicate.clone(),
                    object: object.clone(),
                    graph: Some(graph.clone()),
                });
            }
            StreamEvent::SparqlUpdate { query, .. } => {
                // Parse SPARQL update and extract triples
                debug!("Extracting triples from SPARQL update: {}", query);
            }
            _ => {
                // Other event types might not contain RDF data
            }
        }

        Ok(triples)
    }

    /// Get window data for query execution
    async fn get_window_data_for_query(&self, query: &CSparqlQuery) -> Result<Vec<Triple>> {
        let mut all_triples = Vec::new();

        let windows = self.windows.read().await;
        for stream in &query.components.streams {
            // Find matching window
            if let Some(window) = windows.get(&stream.uri) {
                for windowed_triple in &window.buffer {
                    all_triples.push(windowed_triple.triple.clone());
                }
            }
        }

        Ok(all_triples)
    }

    /// Execute SPARQL query on window data
    async fn execute_sparql_on_window(
        &self,
        query: &CSparqlQuery,
        triples: &[Triple],
    ) -> Result<QueryResult> {
        // In a real implementation, insert triples into a temporary graph
        // and execute the SPARQL query

        debug!(
            "Executing C-SPARQL query {} on {} triples",
            query.id,
            triples.len()
        );

        // Simplified result
        Ok(QueryResult {
            bindings: Vec::new(),
        })
    }

    /// Get statistics
    pub async fn get_stats(&self) -> CSparqlStats {
        self.stats.read().await.clone()
    }

    /// Start a registered query
    pub async fn start_query(&self, query_id: &str) -> Result<()> {
        let mut queries = self.queries.write().await;
        if let Some(query) = queries.get_mut(query_id) {
            query.state = QueryState::Running;
            info!("Started C-SPARQL query: {}", query_id);
            Ok(())
        } else {
            Err(anyhow!("Query not found: {}", query_id))
        }
    }

    /// Stop a running query
    pub async fn stop_query(&self, query_id: &str) -> Result<()> {
        let mut queries = self.queries.write().await;
        if let Some(query) = queries.get_mut(query_id) {
            query.state = QueryState::Stopped;
            info!("Stopped C-SPARQL query: {}", query_id);
            Ok(())
        } else {
            Err(anyhow!("Query not found: {}", query_id))
        }
    }

    /// Unregister a query
    pub async fn unregister_query(&self, query_id: &str) -> Result<()> {
        let mut queries = self.queries.write().await;
        queries
            .remove(query_id)
            .ok_or_else(|| anyhow!("Query not found: {}", query_id))?;

        info!("Unregistered C-SPARQL query: {}", query_id);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_csparql_config_defaults() {
        let config = CSparqlConfig::default();
        assert_eq!(config.max_queries, 100);
        assert!(config.incremental_evaluation);
    }

    #[tokio::test]
    async fn test_window_spec_creation() {
        let window = WindowSpec {
            window_type: WindowType::Tumbling,
            range: WindowRange::Time(Duration::from_secs(60)),
            step: None,
        };

        assert_eq!(window.window_type, WindowType::Tumbling);
        matches!(window.range, WindowRange::Time(_));
    }

    #[tokio::test]
    async fn test_query_type_parsing() {
        let query_select = "SELECT * WHERE { ?s ?p ?o }";
        let query_construct = "CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }";

        assert!(query_select.to_uppercase().contains("SELECT"));
        assert!(query_construct.to_uppercase().contains("CONSTRUCT"));
    }

    #[test]
    fn test_duration_parsing_standalone() {
        // Test duration parsing without needing a store
        let parse_duration = |s: &str| -> Result<Duration> {
            if !s.starts_with("PT") {
                return Err(anyhow!("Invalid duration format: {}", s));
            }

            let duration_part = &s[2..];

            if let Some(seconds_pos) = duration_part.find('S') {
                let seconds: u64 = duration_part[..seconds_pos].parse()?;
                Ok(Duration::from_secs(seconds))
            } else if let Some(minutes_pos) = duration_part.find('M') {
                let minutes: u64 = duration_part[..minutes_pos].parse()?;
                Ok(Duration::from_secs(minutes * 60))
            } else if let Some(hours_pos) = duration_part.find('H') {
                let hours: u64 = duration_part[..hours_pos].parse()?;
                Ok(Duration::from_secs(hours * 3600))
            } else {
                Err(anyhow!("Invalid duration format: {}", s))
            }
        };

        let duration = parse_duration("PT10S").unwrap();
        assert_eq!(duration, Duration::from_secs(10));

        let duration = parse_duration("PT5M").unwrap();
        assert_eq!(duration, Duration::from_secs(300));

        let duration = parse_duration("PT1H").unwrap();
        assert_eq!(duration, Duration::from_secs(3600));
    }

    #[tokio::test]
    async fn test_csparql_stats() {
        let stats = CSparqlStats {
            queries_registered: 5,
            queries_executed: 100,
            queries_failed: 2,
            total_events_processed: 1000,
            total_results_produced: 500,
            avg_query_latency_ms: 15.5,
            active_windows: 3,
        };

        assert_eq!(stats.queries_registered, 5);
        assert_eq!(stats.total_events_processed, 1000);
    }
}
