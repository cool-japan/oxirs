//! # RDF Stream Processing (RSP) Module
//!
//! This module implements C-SPARQL and CQELS extensions for streaming SPARQL queries.
//!
//! ## Supported Features
//!
//! ### C-SPARQL (Continuous SPARQL)
//! - FROM STREAM clause for stream registration
//! - WINDOW operators (RANGE, TRIPLES, TUMBLING, SLIDING)
//! - Time-based windows with configurable ranges
//! - Aggregations over windows
//! - Stream-to-relation operators
//!
//! ### CQELS (Continuous Query Evaluation over Linked Streams)
//! - Native stream processing semantics
//! - Incremental query evaluation
//! - Sliding window processing
//! - Multi-stream joins with window constraints
//!
//! ## Example Queries
//!
//! ```sparql
//! # C-SPARQL: Temperature average over 5-minute sliding window
//! SELECT ?sensor (AVG(?temp) AS ?avgTemp)
//! FROM STREAM <http://sensors.example/stream> [RANGE 5m STEP 1m]
//! WHERE {
//!     ?sensor rdf:type :TemperatureSensor .
//!     ?sensor :temperature ?temp .
//! }
//! GROUP BY ?sensor
//! ```
//!
//! ```sparql
//! # CQELS: Pattern detection with windowing
//! SELECT ?user ?action1 ?action2
//! FROM STREAM <http://events.example/stream> [NOW-5m TO NOW]
//! WHERE {
//!     ?event1 :user ?user ; :action ?action1 ; :timestamp ?t1 .
//!     ?event2 :user ?user ; :action ?action2 ; :timestamp ?t2 .
//!     FILTER(?t2 > ?t1 && ?t2 - ?t1 < 60)
//! }
//! ```

use anyhow::{anyhow, Result};
use chrono::{DateTime, Duration as ChronoDuration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::{
    sparql_streaming::{ContinuousQueryManager, QueryMetadata, QueryResultChannel},
    store_integration::{RdfStore, Triple},
    StreamEvent,
};

/// RSP query language
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RspLanguage {
    /// C-SPARQL (Continuous SPARQL)
    CSparql,
    /// CQELS (Continuous Query Evaluation over Linked Streams)
    Cqels,
    /// Standard SPARQL with streaming extensions
    SparqlStream,
}

impl std::fmt::Display for RspLanguage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RspLanguage::CSparql => write!(f, "C-SPARQL"),
            RspLanguage::Cqels => write!(f, "CQELS"),
            RspLanguage::SparqlStream => write!(f, "SPARQL-Stream"),
        }
    }
}

/// RSP query parser and processor
pub struct RspProcessor {
    /// Registered streams
    streams: Arc<RwLock<HashMap<String, StreamDescriptor>>>,
    /// Active windows
    windows: Arc<RwLock<HashMap<String, WindowManager>>>,
    /// Query manager for execution
    query_manager: Arc<ContinuousQueryManager>,
    /// RDF store
    store: Arc<dyn RdfStore>,
    /// Configuration
    config: RspConfig,
}

/// RSP configuration
#[derive(Debug, Clone)]
pub struct RspConfig {
    /// Default window size
    pub default_window_size: ChronoDuration,
    /// Default window slide
    pub default_window_slide: ChronoDuration,
    /// Maximum window size
    pub max_window_size: ChronoDuration,
    /// Enable incremental evaluation
    pub enable_incremental_eval: bool,
    /// Maximum concurrent windows
    pub max_concurrent_windows: usize,
    /// Window cleanup interval
    pub cleanup_interval: Duration,
}

impl Default for RspConfig {
    fn default() -> Self {
        Self {
            default_window_size: ChronoDuration::minutes(5),
            default_window_slide: ChronoDuration::minutes(1),
            max_window_size: ChronoDuration::hours(24),
            enable_incremental_eval: true,
            max_concurrent_windows: 1000,
            cleanup_interval: Duration::from_secs(60),
        }
    }
}

/// Stream descriptor
#[derive(Debug, Clone)]
pub struct StreamDescriptor {
    /// Stream URI
    pub uri: String,
    /// Stream name
    pub name: String,
    /// Schema (optional)
    pub schema: Option<String>,
    /// Window configuration
    pub window: Option<WindowConfig>,
    /// Stream metadata
    pub metadata: HashMap<String, String>,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
}

/// Window configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowConfig {
    /// Window type
    pub window_type: WindowType,
    /// Window size
    pub size: WindowSize,
    /// Window slide (for sliding windows)
    pub slide: Option<WindowSize>,
    /// Start time (for time-based windows)
    pub start_time: Option<DateTime<Utc>>,
    /// End time (for time-based windows)
    pub end_time: Option<DateTime<Utc>>,
}

/// Window type
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum WindowType {
    /// Tumbling window (non-overlapping)
    Tumbling,
    /// Sliding window (overlapping)
    Sliding,
    /// Landmark window (from start to now)
    Landmark,
    /// Session window (activity-based)
    Session { gap: ChronoDuration },
}

/// Window size specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WindowSize {
    /// Time-based (e.g., 5 minutes)
    Time(ChronoDuration),
    /// Triple count-based (e.g., 1000 triples)
    Triples(usize),
    /// Logical time-based (e.g., 100 events)
    Logical(usize),
}

/// Window manager for a specific stream
pub struct WindowManager {
    /// Stream URI
    stream_uri: String,
    /// Window configuration
    config: WindowConfig,
    /// Current window contents
    windows: VecDeque<Window>,
    /// Statistics
    stats: WindowStats,
}

/// Active window
#[derive(Debug, Clone)]
pub struct Window {
    /// Window ID
    pub id: String,
    /// Window start time
    pub start: DateTime<Utc>,
    /// Window end time
    pub end: DateTime<Utc>,
    /// Triples in window
    pub triples: Vec<Triple>,
    /// Materialized as graph (for optimization)
    pub materialized: bool,
}

/// Window statistics
#[derive(Debug, Clone, Default)]
pub struct WindowStats {
    /// Total windows created
    pub windows_created: u64,
    /// Total windows closed
    pub windows_closed: u64,
    /// Active windows
    pub active_windows: usize,
    /// Total triples processed
    pub triples_processed: u64,
    /// Average window size
    pub avg_window_size: f64,
}

/// Parsed RSP query
#[derive(Debug, Clone)]
pub struct RspQuery {
    /// Query language
    pub language: RspLanguage,
    /// Original query string
    pub original: String,
    /// Parsed stream clauses
    pub streams: Vec<StreamClause>,
    /// Window specifications
    pub windows: Vec<WindowConfig>,
    /// Base SPARQL query (after stream/window extraction)
    pub base_query: String,
    /// Query metadata
    pub metadata: QueryMetadata,
}

/// Stream clause (FROM STREAM)
#[derive(Debug, Clone)]
pub struct StreamClause {
    /// Stream URI
    pub uri: String,
    /// Window specification
    pub window: Option<WindowConfig>,
    /// Named graph (if any)
    pub graph: Option<String>,
}

impl RspProcessor {
    /// Create a new RSP processor
    pub async fn new(
        store: Arc<dyn RdfStore>,
        query_manager: Arc<ContinuousQueryManager>,
        config: RspConfig,
    ) -> Result<Self> {
        Ok(Self {
            streams: Arc::new(RwLock::new(HashMap::new())),
            windows: Arc::new(RwLock::new(HashMap::new())),
            query_manager,
            store,
            config,
        })
    }

    /// Register a stream
    pub async fn register_stream(&self, descriptor: StreamDescriptor) -> Result<()> {
        let uri = descriptor.uri.clone();

        // Create window manager if window config provided
        if let Some(window_config) = &descriptor.window {
            let manager = WindowManager {
                stream_uri: uri.clone(),
                config: window_config.clone(),
                windows: VecDeque::new(),
                stats: WindowStats::default(),
            };

            self.windows.write().await.insert(uri.clone(), manager);
        }

        self.streams.write().await.insert(uri.clone(), descriptor);

        info!("Registered RSP stream: {}", uri);
        Ok(())
    }

    /// Unregister a stream
    pub async fn unregister_stream(&self, uri: &str) -> Result<()> {
        self.streams
            .write()
            .await
            .remove(uri)
            .ok_or_else(|| anyhow!("Stream not found: {}", uri))?;

        self.windows.write().await.remove(uri);

        info!("Unregistered RSP stream: {}", uri);
        Ok(())
    }

    /// Parse an RSP query
    pub fn parse_query(&self, query: &str) -> Result<RspQuery> {
        // Detect query language
        let language = self.detect_language(query)?;

        // Extract stream clauses
        let streams = self.extract_stream_clauses(query)?;

        // Extract window specifications
        let windows = self.extract_windows(query)?;

        // Generate base SPARQL query (remove RSP extensions)
        let base_query = self.generate_base_query(query, &streams, &windows)?;

        Ok(RspQuery {
            language,
            original: query.to_string(),
            streams,
            windows,
            base_query,
            metadata: QueryMetadata::default(),
        })
    }

    /// Register and execute an RSP query
    pub async fn execute_query(&self, query: &str, channel: QueryResultChannel) -> Result<String> {
        // Parse RSP query
        let rsp_query = self.parse_query(query)?;

        // Validate streams exist
        let streams_guard = self.streams.read().await;
        for stream_clause in &rsp_query.streams {
            if !streams_guard.contains_key(&stream_clause.uri) {
                return Err(anyhow!("Stream not registered: {}", stream_clause.uri));
            }
        }
        drop(streams_guard);

        // Register with continuous query manager
        let query_id = self
            .query_manager
            .register_query(rsp_query.base_query.clone(), rsp_query.metadata, channel)
            .await?;

        info!(
            "Registered RSP query ({}): {}",
            rsp_query.language, query_id
        );

        Ok(query_id)
    }

    /// Process a stream event (add to windows)
    pub async fn process_event(&self, stream_uri: &str, event: &StreamEvent) -> Result<()> {
        // Extract triples from event
        let triples = self.extract_triples_from_event(event)?;

        if triples.is_empty() {
            return Ok(());
        }

        // Get window manager for stream
        let mut windows_guard = self.windows.write().await;

        if let Some(manager) = windows_guard.get_mut(stream_uri) {
            // Add triples to appropriate windows
            self.add_to_windows(manager, triples).await?;

            // Update statistics
            manager.stats.triples_processed += 1;

            debug!(
                "Processed event for stream {}: {} active windows",
                stream_uri, manager.stats.active_windows
            );
        }

        Ok(())
    }

    /// Detect RSP query language
    pub fn detect_language(&self, query: &str) -> Result<RspLanguage> {
        Self::detect_language_static(query)
    }

    /// Detect RSP query language (static version for testing)
    fn detect_language_static(query: &str) -> Result<RspLanguage> {
        let query_upper = query.to_uppercase();

        if query_upper.contains("FROM STREAM") || query_upper.contains("[RANGE") {
            Ok(RspLanguage::CSparql)
        } else if query_upper.contains("[NOW-") || query_upper.contains("TO NOW") {
            Ok(RspLanguage::Cqels)
        } else if query_upper.contains("STREAM") || query_upper.contains("WINDOW") {
            Ok(RspLanguage::SparqlStream)
        } else {
            Err(anyhow!("Query does not appear to contain RSP extensions"))
        }
    }

    /// Extract FROM STREAM clauses
    fn extract_stream_clauses(&self, query: &str) -> Result<Vec<StreamClause>> {
        let mut clauses = Vec::new();

        // Simple regex-based extraction (in production, use proper SPARQL parser)
        let lines = query.lines();

        for line in lines {
            if line.to_uppercase().contains("FROM STREAM") {
                // Extract stream URI
                if let Some(start) = line.find('<') {
                    if let Some(end) = line[start..].find('>') {
                        let uri = line[start + 1..start + end].to_string();

                        // Extract window if present
                        let window = if line.contains('[') {
                            self.parse_window_spec(line).ok()
                        } else {
                            None
                        };

                        clauses.push(StreamClause {
                            uri,
                            window,
                            graph: None,
                        });
                    }
                }
            }
        }

        if clauses.is_empty() {
            // Check for implicit streams in WHERE clause
            warn!("No explicit FROM STREAM clauses found");
        }

        Ok(clauses)
    }

    /// Extract window specifications
    fn extract_windows(&self, query: &str) -> Result<Vec<WindowConfig>> {
        let mut windows = Vec::new();

        // Look for window specifications
        for line in query.lines() {
            if line.contains('[') && line.contains(']') {
                if let Ok(window) = self.parse_window_spec(line) {
                    windows.push(window);
                }
            }
        }

        Ok(windows)
    }

    /// Parse window specification
    fn parse_window_spec(&self, spec: &str) -> Result<WindowConfig> {
        let spec_upper = spec.to_uppercase();

        // C-SPARQL: [RANGE 5m STEP 1m]
        if spec_upper.contains("RANGE") {
            let size = self.parse_duration(spec, "RANGE")?;
            let slide = if spec_upper.contains("STEP") {
                Some(self.parse_duration(spec, "STEP")?)
            } else {
                None
            };

            return Ok(WindowConfig {
                window_type: if slide.is_some() {
                    WindowType::Sliding
                } else {
                    WindowType::Tumbling
                },
                size: WindowSize::Time(size),
                slide: slide.map(WindowSize::Time),
                start_time: None,
                end_time: None,
            });
        }

        // CQELS: [NOW-5m TO NOW]
        if spec_upper.contains("NOW-") && spec_upper.contains("TO NOW") {
            if let Some(start_idx) = spec.find("NOW-") {
                let duration_str = &spec[start_idx + 4..];
                if let Some(end_idx) = duration_str.find("TO") {
                    let duration_str = duration_str[..end_idx].trim();
                    let size = self.parse_duration_string(duration_str)?;

                    return Ok(WindowConfig {
                        window_type: WindowType::Sliding,
                        size: WindowSize::Time(size),
                        slide: Some(WindowSize::Time(ChronoDuration::seconds(1))),
                        start_time: Some(Utc::now() - size),
                        end_time: Some(Utc::now()),
                    });
                }
            }
        }

        // Triples-based: [TRIPLES 1000]
        if spec_upper.contains("TRIPLES") {
            if let Some(start) = spec.find("TRIPLES") {
                let num_str = &spec[start + 7..].trim();
                if let Some(end) = num_str.find(|c: char| !c.is_numeric()) {
                    let count: usize = num_str[..end].parse()?;
                    return Ok(WindowConfig {
                        window_type: WindowType::Tumbling,
                        size: WindowSize::Triples(count),
                        slide: None,
                        start_time: None,
                        end_time: None,
                    });
                }
            }
        }

        Err(anyhow!("Unable to parse window specification: {}", spec))
    }

    /// Parse duration from spec
    fn parse_duration(&self, spec: &str, keyword: &str) -> Result<ChronoDuration> {
        if let Some(start) = spec.to_uppercase().find(keyword) {
            let duration_str = &spec[start + keyword.len()..].trim();

            // Extract duration string until next keyword or bracket
            let duration_str = duration_str
                .split_whitespace()
                .next()
                .unwrap_or("")
                .replace(']', "");

            self.parse_duration_string(&duration_str)
        } else {
            Err(anyhow!("Keyword not found: {}", keyword))
        }
    }

    /// Parse duration string (e.g., "5m", "1h", "30s")
    pub fn parse_duration_string(&self, s: &str) -> Result<ChronoDuration> {
        Self::parse_duration_string_static(s)
    }

    /// Parse duration string (static version for testing)
    fn parse_duration_string_static(s: &str) -> Result<ChronoDuration> {
        let s = s.trim().to_lowercase();

        if s.is_empty() {
            return Err(anyhow!("Empty duration string"));
        }

        // Extract number and unit
        let num_end = s.chars().position(|c| !c.is_numeric()).unwrap_or(s.len());
        let num: i64 = s[..num_end].parse()?;
        let unit = &s[num_end..];

        match unit {
            "s" | "sec" | "second" | "seconds" => Ok(ChronoDuration::seconds(num)),
            "m" | "min" | "minute" | "minutes" => Ok(ChronoDuration::minutes(num)),
            "h" | "hr" | "hour" | "hours" => Ok(ChronoDuration::hours(num)),
            "d" | "day" | "days" => Ok(ChronoDuration::days(num)),
            _ => Err(anyhow!("Unknown duration unit: {}", unit)),
        }
    }

    /// Generate base SPARQL query (remove RSP extensions)
    fn generate_base_query(
        &self,
        original: &str,
        streams: &[StreamClause],
        _windows: &[WindowConfig],
    ) -> Result<String> {
        let mut base_query = original.to_string();

        // Remove FROM STREAM clauses
        for stream in streams {
            // Simple replacement (in production, use proper SPARQL rewriting)
            base_query = base_query.replace(&format!("FROM STREAM <{}>", stream.uri), "");

            // Remove window specifications
            if let Some(start) = base_query.find('[') {
                if let Some(end) = base_query[start..].find(']') {
                    base_query.replace_range(start..start + end + 1, "");
                }
            }
        }

        // Clean up extra whitespace
        base_query = base_query
            .lines()
            .filter(|line| !line.trim().is_empty())
            .collect::<Vec<_>>()
            .join("\n");

        Ok(base_query)
    }

    /// Extract triples from stream event
    fn extract_triples_from_event(&self, event: &StreamEvent) -> Result<Vec<Triple>> {
        match event {
            StreamEvent::TripleAdded {
                subject,
                predicate,
                object,
                graph,
                metadata: _,
            } => Ok(vec![Triple {
                subject: subject.clone(),
                predicate: predicate.clone(),
                object: object.clone(),
                graph: graph.clone(),
            }]),
            StreamEvent::QuadAdded {
                subject,
                predicate,
                object,
                graph,
                metadata: _,
            } => Ok(vec![Triple {
                subject: subject.clone(),
                predicate: predicate.clone(),
                object: object.clone(),
                graph: Some(graph.clone()),
            }]),
            _ => Ok(vec![]),
        }
    }

    /// Add triples to appropriate windows
    async fn add_to_windows(
        &self,
        manager: &mut WindowManager,
        triples: Vec<Triple>,
    ) -> Result<()> {
        let now = Utc::now();

        match &manager.config.window_type {
            WindowType::Tumbling => {
                // Create new window if needed
                if manager.windows.is_empty()
                    || self.window_is_full(
                        manager
                            .windows
                            .back()
                            .expect("windows validated to be non-empty via is_empty check"),
                        &manager.config,
                    )
                {
                    self.create_new_window(manager, now).await?;
                }

                // Add to current window
                if let Some(window) = manager.windows.back_mut() {
                    window.triples.extend(triples);
                }
            }
            WindowType::Sliding => {
                // For sliding windows, may need to add to multiple windows
                // and create new windows as time progresses

                // Remove expired windows
                self.cleanup_expired_windows(manager, now);

                // Ensure we have at least one active window
                if manager.windows.is_empty() {
                    self.create_new_window(manager, now).await?;
                }

                // Add triples to all active windows
                for window in &mut manager.windows {
                    if now >= window.start && now <= window.end {
                        window.triples.extend(triples.clone());
                    }
                }

                // Check if we need a new sliding window
                if let Some(WindowSize::Time(slide)) = &manager.config.slide {
                    if let Some(last_window) = manager.windows.back() {
                        let next_start = last_window.start + *slide;
                        if now >= next_start {
                            self.create_new_window(manager, next_start).await?;
                        }
                    }
                }
            }
            WindowType::Landmark => {
                // Landmark windows grow from start to now
                if manager.windows.is_empty() {
                    self.create_new_window(manager, now).await?;
                }

                if let Some(window) = manager.windows.front_mut() {
                    window.triples.extend(triples);
                    window.end = now;
                }
            }
            WindowType::Session { gap } => {
                // Session windows based on activity gaps
                if manager.windows.is_empty() {
                    self.create_new_window(manager, now).await?;
                } else if let Some(last_window) = manager.windows.back_mut() {
                    // Check if within session gap
                    if now - last_window.end <= *gap {
                        // Extend existing session
                        last_window.triples.extend(triples);
                        last_window.end = now;
                    } else {
                        // Start new session
                        self.create_new_window(manager, now).await?;
                    }
                }
            }
        }

        manager.stats.active_windows = manager.windows.len();
        Ok(())
    }

    /// Check if window is full
    fn window_is_full(&self, window: &Window, config: &WindowConfig) -> bool {
        match &config.size {
            WindowSize::Time(duration) => {
                let window_duration = window.end - window.start;
                window_duration >= *duration
            }
            WindowSize::Triples(count) => window.triples.len() >= *count,
            WindowSize::Logical(count) => window.triples.len() >= *count,
        }
    }

    /// Create a new window
    async fn create_new_window(
        &self,
        manager: &mut WindowManager,
        start: DateTime<Utc>,
    ) -> Result<()> {
        let end = match &manager.config.size {
            WindowSize::Time(duration) => start + *duration,
            _ => start, // For count-based, will be updated as triples arrive
        };

        let window = Window {
            id: uuid::Uuid::new_v4().to_string(),
            start,
            end,
            triples: Vec::new(),
            materialized: false,
        };

        manager.windows.push_back(window);
        manager.stats.windows_created += 1;

        Ok(())
    }

    /// Cleanup expired windows
    fn cleanup_expired_windows(&self, manager: &mut WindowManager, now: DateTime<Utc>) {
        while let Some(window) = manager.windows.front() {
            if window.end < now {
                manager.windows.pop_front();
                manager.stats.windows_closed += 1;
            } else {
                break;
            }
        }
    }

    /// Get window statistics
    pub async fn get_window_stats(&self, stream_uri: &str) -> Option<WindowStats> {
        self.windows
            .read()
            .await
            .get(stream_uri)
            .map(|m| m.stats.clone())
    }

    /// List registered streams
    pub async fn list_streams(&self) -> Vec<String> {
        self.streams.read().await.keys().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_duration() {
        assert_eq!(
            RspProcessor::parse_duration_string_static("5m").unwrap(),
            ChronoDuration::minutes(5)
        );
        assert_eq!(
            RspProcessor::parse_duration_string_static("1h").unwrap(),
            ChronoDuration::hours(1)
        );
        assert_eq!(
            RspProcessor::parse_duration_string_static("30s").unwrap(),
            ChronoDuration::seconds(30)
        );
    }

    #[test]
    fn test_detect_language() {
        let csparql_query = "SELECT * FROM STREAM <http://stream> [RANGE 5m]";
        assert_eq!(
            RspProcessor::detect_language_static(csparql_query).unwrap(),
            RspLanguage::CSparql
        );

        let cqels_query = "SELECT * WHERE { ?s ?p ?o } [NOW-5m TO NOW]";
        assert_eq!(
            RspProcessor::detect_language_static(cqels_query).unwrap(),
            RspLanguage::Cqels
        );
    }
}
