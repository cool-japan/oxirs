//! WebSocket Streaming for SPARQL Query Results
//!
//! Provides real-time streaming of SPARQL query results over WebSocket connections.
//! Supports incremental result delivery, query cancellation, and backpressure handling.

use crate::algebra::Variable;
use anyhow::{anyhow, Result};
use scirs2_core::metrics::{Counter, Gauge, Timer};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, RwLock};

/// Configuration for WebSocket streaming
#[derive(Debug, Clone)]
pub struct WebSocketConfig {
    /// Maximum message size in bytes
    pub max_message_size: usize,
    /// Buffer size for streaming results
    pub buffer_size: usize,
    /// Ping interval for keepalive
    pub ping_interval: Duration,
    /// Connection timeout
    pub connection_timeout: Duration,
    /// Maximum concurrent connections
    pub max_connections: usize,
    /// Enable result compression
    pub enable_compression: bool,
    /// Batch size for result streaming
    pub batch_size: usize,
}

impl Default for WebSocketConfig {
    fn default() -> Self {
        Self {
            max_message_size: 16 * 1024 * 1024, // 16 MB
            buffer_size: 10000,
            ping_interval: Duration::from_secs(30),
            connection_timeout: Duration::from_secs(300),
            max_connections: 1000,
            enable_compression: true,
            batch_size: 100,
        }
    }
}

/// WebSocket message types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum WebSocketMessage {
    /// Query request
    Query {
        id: String,
        sparql: String,
        bindings: Option<HashMap<String, String>>,
    },
    /// Query result batch
    ResultBatch {
        id: String,
        variables: Vec<String>,
        solutions: Vec<HashMap<String, String>>,
        more: bool,
    },
    /// Query completion
    QueryComplete { id: String, total_results: usize },
    /// Query error
    QueryError { id: String, error: String },
    /// Query cancellation request
    CancelQuery { id: String },
    /// Query cancelled confirmation
    QueryCancelled { id: String },
    /// Server ping
    Ping,
    /// Client pong
    Pong,
    /// Connection statistics
    Stats { stats: ConnectionStats },
}

/// WebSocket streaming session
pub struct WebSocketSession {
    /// Session ID
    id: String,
    /// Configuration
    config: WebSocketConfig,
    /// Active queries
    active_queries: Arc<RwLock<HashMap<String, QuerySession>>>,
    /// Metrics
    metrics: Arc<SessionMetrics>,
    /// Connection start time
    start_time: Instant,
}

impl WebSocketSession {
    /// Create a new WebSocket session
    pub fn new(id: String, config: WebSocketConfig) -> Self {
        Self {
            id,
            config,
            active_queries: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(SessionMetrics::new()),
            start_time: Instant::now(),
        }
    }

    /// Start a query execution
    pub async fn start_query(
        &self,
        query_id: String,
        sparql: String,
    ) -> Result<mpsc::Receiver<WebSocketMessage>> {
        let (tx, rx) = mpsc::channel(self.config.buffer_size);

        // Create query session
        let session = QuerySession {
            id: query_id.clone(),
            sparql: sparql.clone(),
            start_time: Instant::now(),
            results_sent: 0,
            cancelled: false,
            sender: tx.clone(),
        };

        // Register query
        {
            let mut queries = self.active_queries.write().await;
            if queries.len() >= self.config.max_connections {
                return Err(anyhow!("Maximum concurrent queries reached"));
            }
            queries.insert(query_id.clone(), session);
        }

        self.metrics.active_queries.add(1.0);
        self.metrics.total_queries.inc();

        Ok(rx)
    }

    /// Stream results for a query
    /// Note: Solution is Vec<Binding>, where Binding is HashMap<Variable, Term>
    /// For streaming, we expect bindings (individual solutions), not Solution (Vec<Binding>)
    pub async fn stream_results(
        &self,
        query_id: &str,
        variables: Vec<Variable>,
        bindings: Vec<crate::algebra::Binding>,
    ) -> Result<()> {
        let query_session = {
            let queries = self.active_queries.read().await;
            queries
                .get(query_id)
                .ok_or_else(|| anyhow!("Query not found: {}", query_id))?
                .clone()
        };

        if query_session.cancelled {
            return Ok(());
        }

        // Convert variables to strings
        let var_names: Vec<String> = variables.iter().map(|v| v.to_string()).collect();

        // Stream bindings in batches
        for batch in bindings.chunks(self.config.batch_size) {
            if query_session.is_cancelled() {
                break;
            }

            // Convert bindings to string maps
            let solution_maps: Vec<HashMap<String, String>> = batch
                .iter()
                .map(|binding| {
                    binding
                        .iter()
                        .map(|(var, term)| (var.to_string(), format!("{:?}", term)))
                        .collect()
                })
                .collect();

            let message = WebSocketMessage::ResultBatch {
                id: query_id.to_string(),
                variables: var_names.clone(),
                solutions: solution_maps,
                more: true,
            };

            query_session
                .sender
                .send(message)
                .await
                .map_err(|e| anyhow!("Failed to send results: {}", e))?;

            // Update metrics
            self.metrics.results_sent.add(batch.len() as u64);

            // Update query session
            {
                let mut queries = self.active_queries.write().await;
                if let Some(session) = queries.get_mut(query_id) {
                    session.results_sent += batch.len();
                }
            }
        }

        // Send completion message
        let message = WebSocketMessage::QueryComplete {
            id: query_id.to_string(),
            total_results: bindings.len(),
        };

        query_session
            .sender
            .send(message)
            .await
            .map_err(|e| anyhow!("Failed to send completion: {}", e))?;

        // Clean up query
        self.complete_query(query_id).await;

        Ok(())
    }

    /// Cancel a query
    pub async fn cancel_query(&self, query_id: &str) -> Result<()> {
        let mut queries = self.active_queries.write().await;
        if let Some(session) = queries.get_mut(query_id) {
            session.cancelled = true;

            let message = WebSocketMessage::QueryCancelled {
                id: query_id.to_string(),
            };

            let _ = session.sender.send(message).await;

            self.metrics.queries_cancelled.inc();
        }

        queries.remove(query_id);
        self.metrics.active_queries.sub(1.0);

        Ok(())
    }

    /// Complete a query
    async fn complete_query(&self, query_id: &str) {
        let mut queries = self.active_queries.write().await;
        if let Some(session) = queries.remove(query_id) {
            let duration = session.start_time.elapsed();
            self.metrics.query_duration.observe(duration);
            self.metrics.active_queries.sub(1.0);
            self.metrics.completed_queries.inc();
        }
    }

    /// Send error to query
    pub async fn send_error(&self, query_id: &str, error: String) -> Result<()> {
        let queries = self.active_queries.read().await;
        if let Some(session) = queries.get(query_id) {
            let message = WebSocketMessage::QueryError {
                id: query_id.to_string(),
                error,
            };

            let _ = session.sender.send(message).await;
            self.metrics.query_errors.inc();
        }

        Ok(())
    }

    /// Get session statistics
    pub async fn statistics(&self) -> ConnectionStats {
        let queries = self.active_queries.read().await;
        let stats = self.metrics.query_duration.get_stats();

        ConnectionStats {
            session_id: self.id.clone(),
            uptime: self.start_time.elapsed(),
            active_queries: queries.len(),
            total_queries: self.metrics.total_queries.get(),
            completed_queries: self.metrics.completed_queries.get(),
            cancelled_queries: self.metrics.queries_cancelled.get(),
            failed_queries: self.metrics.query_errors.get(),
            results_sent: self.metrics.results_sent.get(),
            average_query_duration: stats.mean,
        }
    }

    /// Check if session is healthy
    pub fn is_healthy(&self) -> bool {
        self.start_time.elapsed() < self.config.connection_timeout
    }
}

/// Query execution session
#[derive(Clone)]
#[allow(dead_code)]
struct QuerySession {
    id: String,
    sparql: String,
    start_time: Instant,
    results_sent: usize,
    cancelled: bool,
    sender: mpsc::Sender<WebSocketMessage>,
}

impl QuerySession {
    fn is_cancelled(&self) -> bool {
        self.cancelled
    }
}

/// Session metrics
struct SessionMetrics {
    total_queries: Counter,
    active_queries: Gauge,
    completed_queries: Counter,
    queries_cancelled: Counter,
    query_errors: Counter,
    results_sent: Counter,
    query_duration: Timer,
}

impl SessionMetrics {
    fn new() -> Self {
        Self {
            total_queries: Counter::new("websocket.total_queries".to_string()),
            active_queries: Gauge::new("websocket.active_queries".to_string()),
            completed_queries: Counter::new("websocket.completed_queries".to_string()),
            queries_cancelled: Counter::new("websocket.queries_cancelled".to_string()),
            query_errors: Counter::new("websocket.query_errors".to_string()),
            results_sent: Counter::new("websocket.results_sent".to_string()),
            query_duration: Timer::new("websocket.query_duration".to_string()),
        }
    }
}

/// Connection statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionStats {
    pub session_id: String,
    #[serde(serialize_with = "serialize_duration")]
    pub uptime: Duration,
    pub active_queries: usize,
    pub total_queries: u64,
    pub completed_queries: u64,
    pub cancelled_queries: u64,
    pub failed_queries: u64,
    pub results_sent: u64,
    pub average_query_duration: f64,
}

fn serialize_duration<S>(duration: &Duration, serializer: S) -> std::result::Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    serializer.serialize_f64(duration.as_secs_f64())
}

/// WebSocket session manager
pub struct WebSocketManager {
    /// Configuration
    config: WebSocketConfig,
    /// Active sessions
    sessions: Arc<RwLock<HashMap<String, Arc<WebSocketSession>>>>,
    /// Global metrics
    metrics: Arc<ManagerMetrics>,
}

impl WebSocketManager {
    /// Create a new WebSocket manager
    pub fn new(config: WebSocketConfig) -> Self {
        Self {
            config,
            sessions: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(ManagerMetrics::new()),
        }
    }

    /// Create a new session
    pub async fn create_session(&self, session_id: String) -> Result<Arc<WebSocketSession>> {
        let mut sessions = self.sessions.write().await;

        if sessions.len() >= self.config.max_connections {
            return Err(anyhow!("Maximum connections reached"));
        }

        let session = Arc::new(WebSocketSession::new(
            session_id.clone(),
            self.config.clone(),
        ));
        sessions.insert(session_id, session.clone());

        self.metrics.active_sessions.add(1.0);
        self.metrics.total_sessions.inc();

        Ok(session)
    }

    /// Get a session
    pub async fn get_session(&self, session_id: &str) -> Option<Arc<WebSocketSession>> {
        let sessions = self.sessions.read().await;
        sessions.get(session_id).cloned()
    }

    /// Remove a session
    pub async fn remove_session(&self, session_id: &str) -> Result<()> {
        let mut sessions = self.sessions.write().await;
        if sessions.remove(session_id).is_some() {
            self.metrics.active_sessions.sub(1.0);
            self.metrics.closed_sessions.inc();
        }
        Ok(())
    }

    /// Get manager statistics
    pub async fn statistics(&self) -> ManagerStats {
        let sessions = self.sessions.read().await;

        ManagerStats {
            active_sessions: sessions.len(),
            total_sessions: self.metrics.total_sessions.get(),
            closed_sessions: self.metrics.closed_sessions.get(),
            max_connections: self.config.max_connections,
        }
    }

    /// Clean up inactive sessions
    pub async fn cleanup_inactive_sessions(&self) -> usize {
        let mut sessions = self.sessions.write().await;
        let mut removed = 0;

        sessions.retain(|_, session| {
            if !session.is_healthy() {
                removed += 1;
                false
            } else {
                true
            }
        });

        if removed > 0 {
            self.metrics.active_sessions.sub(removed as f64);
            self.metrics.closed_sessions.add(removed as u64);
        }

        removed
    }
}

/// Manager metrics
struct ManagerMetrics {
    total_sessions: Counter,
    active_sessions: Gauge,
    closed_sessions: Counter,
}

impl ManagerMetrics {
    fn new() -> Self {
        Self {
            total_sessions: Counter::new("websocket.manager.total_sessions".to_string()),
            active_sessions: Gauge::new("websocket.manager.active_sessions".to_string()),
            closed_sessions: Counter::new("websocket.manager.closed_sessions".to_string()),
        }
    }
}

/// Manager statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManagerStats {
    pub active_sessions: usize,
    pub total_sessions: u64,
    pub closed_sessions: u64,
    pub max_connections: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_websocket_session_creation() {
        let config = WebSocketConfig::default();
        let session = WebSocketSession::new("test-session".to_string(), config);
        assert_eq!(session.id, "test-session");
        assert!(session.is_healthy());
    }

    #[tokio::test]
    async fn test_query_lifecycle() {
        let config = WebSocketConfig::default();
        let session = WebSocketSession::new("test-session".to_string(), config);

        // Start query
        let mut rx = session
            .start_query("q1".to_string(), "SELECT * WHERE { ?s ?p ?o }".to_string())
            .await
            .unwrap();

        // Stream results
        let variables = vec![
            Variable::new("s").unwrap(),
            Variable::new("p").unwrap(),
            Variable::new("o").unwrap(),
        ];
        let bindings = vec![]; // Empty results for test

        // Stream in background
        let session_arc = Arc::new(session);
        let session_ref = Arc::clone(&session_arc);
        tokio::spawn(async move {
            session_ref
                .stream_results("q1", variables, bindings)
                .await
                .unwrap();
        });

        // Receive completion
        let msg = rx.recv().await.unwrap();
        match msg {
            WebSocketMessage::QueryComplete { id, total_results } => {
                assert_eq!(id, "q1");
                assert_eq!(total_results, 0);
            }
            _ => panic!("Expected QueryComplete message"),
        }
    }

    #[tokio::test]
    async fn test_query_cancellation() {
        let config = WebSocketConfig::default();
        let session = WebSocketSession::new("test-session".to_string(), config);

        // Start query
        let _rx = session
            .start_query("q1".to_string(), "SELECT * WHERE { ?s ?p ?o }".to_string())
            .await
            .unwrap();

        // Cancel query
        session.cancel_query("q1").await.unwrap();

        // Verify query is removed
        let queries = session.active_queries.read().await;
        assert!(!queries.contains_key("q1"));
    }

    #[tokio::test]
    async fn test_manager() {
        let config = WebSocketConfig::default();
        let manager = WebSocketManager::new(config);

        // Create session
        let session = manager.create_session("s1".to_string()).await.unwrap();
        assert_eq!(session.id, "s1");

        // Get session
        let retrieved = manager.get_session("s1").await.unwrap();
        assert_eq!(retrieved.id, "s1");

        // Get stats
        let stats = manager.statistics().await;
        assert_eq!(stats.active_sessions, 1);
        assert_eq!(stats.total_sessions, 1);

        // Remove session
        manager.remove_session("s1").await.unwrap();
        let stats = manager.statistics().await;
        assert_eq!(stats.active_sessions, 0);
    }
}
