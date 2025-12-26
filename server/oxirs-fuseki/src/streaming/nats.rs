//! NATS integration for lightweight messaging

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::{
    error::FusekiResult,
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

/// NATS producer implementation
pub struct NatsProducer {
    config: NatsConfig,
    // Future enhancement: Add actual NATS client (requires async-nats crate).
    // For 0.1.0-rc.1: Stub implementation provides API surface for future NATS integration.
}

impl NatsProducer {
    /// Create a new NATS producer
    pub async fn new(config: NatsConfig) -> FusekiResult<Self> {
        tracing::info!("Creating NATS producer for: {}", config.url);

        // Future enhancement: Initialize async-nats client with config.
        // For 0.1.0-rc.1: Stub allows testing of streaming pipeline without NATS dependency.
        Ok(Self { config })
    }
}

#[async_trait]
impl StreamProducer for NatsProducer {
    async fn send(&self, event: RDFEvent) -> FusekiResult<()> {
        tracing::debug!("Sending RDF event to NATS");

        // Future enhancement: Implement async-nats publish.
        // For 0.1.0-rc.1: Logs events for debugging. Full NATS integration is optional.
        tracing::info!("Would send to NATS: {:?}", event);

        Ok(())
    }

    async fn send_batch(&self, events: Vec<RDFEvent>) -> FusekiResult<()> {
        tracing::debug!("Sending batch of {} RDF events to NATS", events.len());

        // Future enhancement: Implement async-nats batch publishing.
        // For 0.1.0-rc.1: Logs individual events. Batch optimization is future work.
        for event in events {
            self.send(event).await?;
        }

        Ok(())
    }

    async fn flush(&self) -> FusekiResult<()> {
        tracing::debug!("Flushing NATS producer");
        // Future enhancement: Implement async-nats flush for guaranteed delivery.
        // For 0.1.0-rc.1: No-op. Full NATS integration is optional.
        Ok(())
    }
}

/// NATS consumer implementation
pub struct NatsConsumer {
    config: NatsConfig,
    // Future enhancement: Add actual NATS client (requires async-nats crate).
    // For 0.1.0-rc.1: Stub implementation provides API surface for future NATS integration.
}

impl NatsConsumer {
    /// Create a new NATS consumer
    pub async fn new(config: NatsConfig) -> FusekiResult<Self> {
        tracing::info!("Creating NATS consumer for: {}", config.url);

        // Future enhancement: Initialize async-nats client with config.
        // For 0.1.0-rc.1: Stub allows testing of streaming pipeline without NATS dependency.
        Ok(Self { config })
    }
}

#[async_trait]
impl StreamConsumer for NatsConsumer {
    async fn subscribe(
        &self,
        _handler: Box<dyn crate::streaming::EventHandler>,
    ) -> FusekiResult<()> {
        tracing::info!("Subscribing to NATS events with handler");

        // Future enhancement: Implement async-nats subscribe with subject patterns.
        // For 0.1.0-rc.1: No-op. Full NATS integration is optional.
        Ok(())
    }

    async fn unsubscribe(&self) -> FusekiResult<()> {
        tracing::info!("Unsubscribing from NATS events");

        // Future enhancement: Implement async-nats unsubscribe.
        // For 0.1.0-rc.1: No-op. Full NATS integration is optional.
        Ok(())
    }

    async fn commit(&self) -> FusekiResult<()> {
        tracing::debug!("Committing NATS consumer (no-op for NATS)");
        // NATS doesn't need commit
        Ok(())
    }
}
