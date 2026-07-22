//! NATS integration for lightweight messaging.
//!
//! This is a real, dependency-free implementation of the NATS *core* client
//! protocol (a simple line-oriented text protocol over TCP): it opens a TCP
//! connection, performs the `INFO`/`CONNECT` handshake (with optional
//! token/user-password auth), and publishes events with real `PUB` frames.
//! There is no silent no-op path: a connection or write failure surfaces as a
//! typed [`FusekiError`], so an operator who configures NATS streaming either
//! gets events delivered to their subject or an explicit error — never a silent
//! black hole.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::tcp::{OwnedReadHalf, OwnedWriteHalf};
use tokio::net::TcpStream;
use tokio::sync::Mutex;

use crate::{
    error::{FusekiError, FusekiResult},
    streaming::{RDFEvent, StreamConsumer, StreamProducer},
};

/// NATS-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NatsConfig {
    /// NATS server URL
    pub url: String,
    /// Subject prefix for RDF events
    pub subject_prefix: String,
    /// Authentication token
    pub token: Option<String>,
    /// Username for authentication
    pub username: Option<String>,
    /// Password for authentication
    pub password: Option<String>,
}

impl Default for NatsConfig {
    fn default() -> Self {
        Self {
            url: "nats://localhost:4222".to_string(),
            subject_prefix: "rdf".to_string(),
            token: None,
            username: None,
            password: None,
        }
    }
}

impl From<crate::streaming::NatsConfig> for NatsConfig {
    fn from(config: crate::streaming::NatsConfig) -> Self {
        let url = config
            .servers
            .first()
            .map(|u| u.to_string())
            .unwrap_or_else(|| "nats://localhost:4222".to_string());

        let (username, password, token) = match config.auth {
            Some(crate::streaming::NatsAuth::UserPass { username, password }) => {
                (Some(username), Some(password), None)
            }
            Some(crate::streaming::NatsAuth::Token(token)) => (None, None, Some(token)),
            Some(crate::streaming::NatsAuth::NKey { .. }) => {
                // NKey not supported in simple config, fallback to no auth
                (None, None, None)
            }
            None => (None, None, None),
        };

        Self {
            url,
            subject_prefix: config.subject_prefix,
            token,
            username,
            password,
        }
    }
}

/// Resolve the `host:port` endpoint from a NATS URL, defaulting the port to
/// 4222 when omitted. Accepts `nats://`, `tcp://`, or a bare `host[:port]`, and
/// tolerates an optional `user:pass@` credential prefix (auth is negotiated in
/// the `CONNECT` frame, not the URL).
pub(crate) fn parse_nats_endpoint(url: &str) -> FusekiResult<String> {
    let trimmed = url.trim();
    let without_scheme = trimmed
        .strip_prefix("nats://")
        .or_else(|| trimmed.strip_prefix("tcp://"))
        .unwrap_or(trimmed);
    // Drop any trailing path and any leading `user:pass@` credential segment.
    let host_and_path = without_scheme.split('/').next().unwrap_or(without_scheme);
    let host_port = host_and_path
        .rsplit('@')
        .next()
        .unwrap_or(host_and_path)
        .trim();
    if host_port.is_empty() {
        return Err(FusekiError::configuration(format!(
            "invalid NATS url (no host): {url}"
        )));
    }
    if host_port.contains(':') {
        Ok(host_port.to_string())
    } else {
        Ok(format!("{host_port}:4222"))
    }
}

/// Build the JSON `CONNECT` options line for the handshake, injecting whichever
/// auth mechanism the config supplies.
fn build_connect_line(config: &NatsConfig) -> String {
    let mut options = serde_json::json!({
        "verbose": false,
        "pedantic": false,
        "tls_required": false,
        "name": "oxirs-fuseki",
        "lang": "rust",
        "protocol": 1,
        "echo": true,
    });
    if let Some(token) = &config.token {
        options["auth_token"] = serde_json::json!(token);
    }
    if let Some(user) = &config.username {
        options["user"] = serde_json::json!(user);
    }
    if let Some(pass) = &config.password {
        options["pass"] = serde_json::json!(pass);
    }
    format!("CONNECT {options}\r\n")
}

/// Derive the publish subject for an event: `<prefix>.<event-kind>`.
fn subject_for(prefix: &str, event: &RDFEvent) -> String {
    let kind = match event {
        RDFEvent::TripleAdded { .. } => "triple.added",
        RDFEvent::TripleRemoved { .. } => "triple.removed",
        RDFEvent::QuadAdded { .. } => "quad.added",
        RDFEvent::QuadRemoved { .. } => "quad.removed",
        RDFEvent::GraphCleared { .. } => "graph.cleared",
        RDFEvent::Transaction { .. } => "transaction",
    };
    if prefix.is_empty() {
        kind.to_string()
    } else {
        format!("{prefix}.{kind}")
    }
}

/// Open a NATS core connection: connect TCP, read the server `INFO`, send
/// `CONNECT` + `PING`, and wait for the `PONG` that confirms the handshake (and
/// surfaces an authentication `-ERR` as a typed error).
async fn connect(config: &NatsConfig) -> FusekiResult<(OwnedWriteHalf, BufReader<OwnedReadHalf>)> {
    let endpoint = parse_nats_endpoint(&config.url)?;
    let stream = TcpStream::connect(&endpoint).await.map_err(|e| {
        FusekiError::service_unavailable(format!("NATS connect to {endpoint} failed: {e}"))
    })?;
    let (read_half, mut write_half) = stream.into_split();
    let mut reader = BufReader::new(read_half);

    // The server greets with a single `INFO {json}` line.
    let mut info_line = String::new();
    reader.read_line(&mut info_line).await.map_err(|e| {
        FusekiError::service_unavailable(format!("NATS handshake read failed: {e}"))
    })?;
    if !info_line.starts_with("INFO") {
        return Err(FusekiError::service_unavailable(format!(
            "NATS server did not send INFO greeting (got: {})",
            info_line.trim()
        )));
    }

    // Send CONNECT then PING; a well-formed server replies PONG, an auth failure
    // replies `-ERR`.
    let mut handshake = build_connect_line(config);
    handshake.push_str("PING\r\n");
    write_half
        .write_all(handshake.as_bytes())
        .await
        .map_err(|e| FusekiError::service_unavailable(format!("NATS CONNECT write failed: {e}")))?;
    write_half
        .flush()
        .await
        .map_err(|e| FusekiError::service_unavailable(format!("NATS CONNECT flush failed: {e}")))?;

    // Read lines until we see PONG (skipping any +OK), turning -ERR into an error.
    loop {
        let mut line = String::new();
        let n = reader.read_line(&mut line).await.map_err(|e| {
            FusekiError::service_unavailable(format!("NATS handshake read failed: {e}"))
        })?;
        if n == 0 {
            return Err(FusekiError::service_unavailable(
                "NATS connection closed during handshake".to_string(),
            ));
        }
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with("+OK") {
            continue;
        }
        if trimmed.starts_with("PONG") {
            break;
        }
        if trimmed.starts_with("-ERR") {
            return Err(FusekiError::service_unavailable(format!(
                "NATS handshake rejected: {trimmed}"
            )));
        }
        // Ignore any other protocol lines (e.g. a stray PING from the server,
        // which we answer below) and keep waiting for PONG.
        if trimmed.starts_with("PING") {
            write_half.write_all(b"PONG\r\n").await.map_err(|e| {
                FusekiError::service_unavailable(format!("NATS PONG write failed: {e}"))
            })?;
        }
    }

    Ok((write_half, reader))
}

/// NATS producer implementation backed by a live TCP connection.
pub struct NatsProducer {
    config: NatsConfig,
    /// Serialized access to the write half of the connection.
    writer: Arc<Mutex<OwnedWriteHalf>>,
    /// Background task that answers server `PING`s so the connection stays alive.
    reader_task: tokio::task::JoinHandle<()>,
}

impl NatsProducer {
    /// Create a new NATS producer, establishing the connection eagerly so a
    /// misconfigured broker fails loudly at startup rather than at first send.
    pub async fn new(config: NatsConfig) -> FusekiResult<Self> {
        tracing::info!("Connecting NATS producer to: {}", config.url);
        let (write_half, mut reader) = connect(&config).await?;
        let writer = Arc::new(Mutex::new(write_half));

        // Keep-alive task: respond to server PINGs so an idle producer connection
        // is not dropped by the broker.
        let reader_writer = Arc::clone(&writer);
        let reader_task = tokio::spawn(async move {
            loop {
                let mut line = String::new();
                match reader.read_line(&mut line).await {
                    Ok(0) => {
                        tracing::warn!("NATS producer connection closed by server");
                        break;
                    }
                    Ok(_) => {
                        if line.trim_start().starts_with("PING") {
                            let mut w = reader_writer.lock().await;
                            if w.write_all(b"PONG\r\n").await.is_err() {
                                break;
                            }
                            let _ = w.flush().await;
                        }
                    }
                    Err(e) => {
                        tracing::warn!("NATS producer read loop ended: {e}");
                        break;
                    }
                }
            }
        });

        Ok(Self {
            config,
            writer,
            reader_task,
        })
    }

    /// Write a single `PUB` frame for one event.
    async fn publish(&self, writer: &mut OwnedWriteHalf, event: &RDFEvent) -> FusekiResult<()> {
        let subject = subject_for(&self.config.subject_prefix, event);
        let payload = serde_json::to_vec(event).map_err(|e| {
            FusekiError::internal(format!("failed to serialize RDF event for NATS: {e}"))
        })?;
        let header = format!("PUB {subject} {}\r\n", payload.len());
        writer
            .write_all(header.as_bytes())
            .await
            .map_err(|e| FusekiError::service_unavailable(format!("NATS PUB write failed: {e}")))?;
        writer.write_all(&payload).await.map_err(|e| {
            FusekiError::service_unavailable(format!("NATS PUB payload write failed: {e}"))
        })?;
        writer
            .write_all(b"\r\n")
            .await
            .map_err(|e| FusekiError::service_unavailable(format!("NATS PUB write failed: {e}")))?;
        Ok(())
    }
}

impl Drop for NatsProducer {
    fn drop(&mut self) {
        self.reader_task.abort();
    }
}

#[async_trait]
impl StreamProducer for NatsProducer {
    async fn send(&self, event: RDFEvent) -> FusekiResult<()> {
        let mut writer = self.writer.lock().await;
        self.publish(&mut writer, &event).await?;
        writer
            .flush()
            .await
            .map_err(|e| FusekiError::service_unavailable(format!("NATS flush failed: {e}")))?;
        Ok(())
    }

    async fn send_batch(&self, events: Vec<RDFEvent>) -> FusekiResult<()> {
        if events.is_empty() {
            return Ok(());
        }
        let mut writer = self.writer.lock().await;
        for event in &events {
            self.publish(&mut writer, event).await?;
        }
        writer
            .flush()
            .await
            .map_err(|e| FusekiError::service_unavailable(format!("NATS flush failed: {e}")))?;
        Ok(())
    }

    async fn flush(&self) -> FusekiResult<()> {
        let mut writer = self.writer.lock().await;
        writer
            .flush()
            .await
            .map_err(|e| FusekiError::service_unavailable(format!("NATS flush failed: {e}")))
    }
}

/// NATS consumer implementation backed by a live TCP connection.
pub struct NatsConsumer {
    config: NatsConfig,
    /// Serialized access to the write half of the connection.
    writer: Arc<Mutex<OwnedWriteHalf>>,
    /// Reader half, taken by [`subscribe`] to drive the dispatch loop.
    reader: Mutex<Option<BufReader<OwnedReadHalf>>>,
    /// Handle to the dispatch loop, aborted on drop / unsubscribe.
    dispatch_task: Mutex<Option<tokio::task::JoinHandle<()>>>,
}

impl NatsConsumer {
    /// Create a new NATS consumer, establishing the connection eagerly.
    pub async fn new(config: NatsConfig) -> FusekiResult<Self> {
        tracing::info!("Connecting NATS consumer to: {}", config.url);
        let (write_half, reader) = connect(&config).await?;
        Ok(Self {
            config,
            writer: Arc::new(Mutex::new(write_half)),
            reader: Mutex::new(Some(reader)),
            dispatch_task: Mutex::new(None),
        })
    }
}

impl Drop for NatsConsumer {
    fn drop(&mut self) {
        if let Ok(mut guard) = self.dispatch_task.try_lock() {
            if let Some(task) = guard.take() {
                task.abort();
            }
        }
    }
}

#[async_trait]
impl StreamConsumer for NatsConsumer {
    async fn subscribe(
        &self,
        handler: Box<dyn crate::streaming::EventHandler>,
    ) -> FusekiResult<()> {
        // Subscribe to every event kind under the configured prefix.
        let subject = if self.config.subject_prefix.is_empty() {
            ">".to_string()
        } else {
            format!("{}.>", self.config.subject_prefix)
        };
        {
            let mut writer = self.writer.lock().await;
            let sub = format!("SUB {subject} 1\r\n");
            writer.write_all(sub.as_bytes()).await.map_err(|e| {
                FusekiError::service_unavailable(format!("NATS SUB write failed: {e}"))
            })?;
            writer.flush().await.map_err(|e| {
                FusekiError::service_unavailable(format!("NATS SUB flush failed: {e}"))
            })?;
        }

        let mut reader = self.reader.lock().await.take().ok_or_else(|| {
            FusekiError::internal("NATS consumer already has an active subscription".to_string())
        })?;
        let writer = Arc::clone(&self.writer);

        let task = tokio::spawn(async move {
            loop {
                let mut line = String::new();
                match reader.read_line(&mut line).await {
                    Ok(0) => break,
                    Ok(_) => {}
                    Err(e) => {
                        tracing::warn!("NATS consumer read failed: {e}");
                        break;
                    }
                }
                let trimmed = line.trim_end();
                if trimmed.starts_with("PING") {
                    let mut w = writer.lock().await;
                    if w.write_all(b"PONG\r\n").await.is_err() {
                        break;
                    }
                    let _ = w.flush().await;
                    continue;
                }
                // `MSG <subject> <sid> [reply] <#bytes>\r\n<payload>\r\n`
                if let Some(rest) = trimmed.strip_prefix("MSG ") {
                    let parts: Vec<&str> = rest.split_whitespace().collect();
                    let byte_count: usize = match parts.last().and_then(|n| n.parse().ok()) {
                        Some(n) => n,
                        None => continue,
                    };
                    let mut payload = vec![0u8; byte_count];
                    if tokio::io::AsyncReadExt::read_exact(&mut reader, &mut payload)
                        .await
                        .is_err()
                    {
                        break;
                    }
                    // Consume the trailing CRLF after the payload.
                    let mut crlf = String::new();
                    let _ = reader.read_line(&mut crlf).await;
                    match serde_json::from_slice::<RDFEvent>(&payload) {
                        Ok(event) => {
                            if let Err(e) = handler.handle(event).await {
                                handler
                                    .on_error(Box::new(std::io::Error::other(e.to_string())))
                                    .await;
                            }
                        }
                        Err(e) => {
                            tracing::warn!("NATS consumer received undecodable event: {e}");
                        }
                    }
                }
            }
        });

        *self.dispatch_task.lock().await = Some(task);
        Ok(())
    }

    async fn unsubscribe(&self) -> FusekiResult<()> {
        if let Some(task) = self.dispatch_task.lock().await.take() {
            task.abort();
        }
        Ok(())
    }

    async fn commit(&self) -> FusekiResult<()> {
        // NATS core has no offset/ack model to commit.
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regression_parse_nats_endpoint_defaults_and_schemes() {
        assert_eq!(
            parse_nats_endpoint("nats://localhost:4222").unwrap(),
            "localhost:4222"
        );
        // Port defaulted to 4222 when absent.
        assert_eq!(
            parse_nats_endpoint("nats://example.com").unwrap(),
            "example.com:4222"
        );
        // Bare host:port and tcp scheme.
        assert_eq!(
            parse_nats_endpoint("10.0.0.5:6222").unwrap(),
            "10.0.0.5:6222"
        );
        assert_eq!(
            parse_nats_endpoint("tcp://broker:4333").unwrap(),
            "broker:4333"
        );
        // Credential prefix stripped.
        assert_eq!(
            parse_nats_endpoint("nats://user:pass@broker:4222").unwrap(),
            "broker:4222"
        );
        assert!(parse_nats_endpoint("nats://").is_err());
    }

    #[test]
    fn regression_subject_for_maps_event_kinds() {
        let ev = RDFEvent::GraphCleared {
            graph: "g".to_string(),
            timestamp: 0,
        };
        assert_eq!(subject_for("rdf", &ev), "rdf.graph.cleared");
        assert_eq!(subject_for("", &ev), "graph.cleared");
    }

    #[tokio::test]
    async fn regression_nats_producer_fails_loud_on_bad_broker() {
        // A producer pointed at a closed port must fail loudly at construction,
        // never silently no-op.
        let config = NatsConfig {
            url: "nats://127.0.0.1:1".to_string(),
            ..Default::default()
        };
        assert!(NatsProducer::new(config).await.is_err());
    }
}
