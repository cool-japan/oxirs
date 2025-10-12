//! NATS Consumer Implementation
//!
//! This module contains the consumer implementation for NATS streaming backend.

use super::config::*;
use super::message::NatsEventMessage;
use crate::error::{StreamError, StreamResult};
use crate::{StreamConfig, StreamEvent};
use anyhow::{anyhow, Result};
use futures_util::StreamExt;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{debug, error, info};

#[cfg(feature = "nats")]
use async_nats::{
    jetstream::{self, consumer::PullConsumer},
    Client, ConnectOptions,
};

/// Consumer statistics for monitoring
#[derive(Debug, Clone, Default)]
pub struct ConsumerStats {
    pub events_consumed: u64,
    pub events_failed: u64,
    pub bytes_received: u64,
    pub last_consume: Option<chrono::DateTime<chrono::Utc>>,
}

/// NATS Consumer for streaming RDF events
pub struct NatsConsumer {
    pub config: StreamConfig,
    pub nats_config: NatsConfig,
    #[cfg(feature = "nats")]
    pub client: Option<Client>,
    #[cfg(feature = "nats")]
    pub jetstream: Option<jetstream::Context>,
    #[cfg(feature = "nats")]
    pub consumer: Option<PullConsumer>,
    #[cfg(not(feature = "nats"))]
    pub _phantom: std::marker::PhantomData<()>,
    pub stats: Arc<RwLock<ConsumerStats>>,
}

impl NatsConsumer {
    pub fn new(config: StreamConfig) -> Result<Self> {
        let nats_config = {
            #[cfg(feature = "nats")]
            {
                if let crate::StreamBackendType::Nats { url, .. } = &config.backend {
                    NatsConfig {
                        url: url.clone(),
                        ..Default::default()
                    }
                } else {
                    NatsConfig::default()
                }
            }
            #[cfg(not(feature = "nats"))]
            {
                NatsConfig::default()
            }
        };

        Ok(Self {
            config,
            nats_config,
            #[cfg(feature = "nats")]
            client: None,
            #[cfg(feature = "nats")]
            jetstream: None,
            #[cfg(feature = "nats")]
            consumer: None,
            #[cfg(not(feature = "nats"))]
            _phantom: std::marker::PhantomData,
            stats: Arc::new(RwLock::new(ConsumerStats::default())),
        })
    }

    pub fn with_nats_config(mut self, nats_config: NatsConfig) -> Self {
        self.nats_config = nats_config;
        self
    }

    /// Apply authentication configuration
    #[cfg(feature = "nats")]
    async fn apply_auth_config(
        &self,
        mut options: ConnectOptions,
        auth: &NatsAuthConfig,
    ) -> Result<ConnectOptions> {
        if let Some(ref token) = auth.token {
            options = options.token(token.clone());
        }
        if let (Some(ref username), Some(ref password)) = (&auth.username, &auth.password) {
            options = options.user_and_password(username.clone(), password.clone());
        }
        if let Some(ref nkey) = auth.nkey {
            options = options.nkey(nkey.clone());
        }
        if let Some(ref creds_file) = auth.creds_file {
            options = options.credentials_file(creds_file).await?;
        }
        Ok(options)
    }

    /// Apply TLS configuration
    #[cfg(feature = "nats")]
    fn apply_tls_config(
        &self,
        mut options: ConnectOptions,
        tls: &NatsTlsConfig,
    ) -> Result<ConnectOptions> {
        if tls.verify_cert {
            options = options.require_tls(true);
        }
        Ok(options)
    }

    #[cfg(feature = "nats")]
    pub async fn connect(&mut self) -> Result<()> {
        // Build connection options with cluster support
        let mut connect_options = ConnectOptions::new()
            .name("oxirs-nats-consumer")
            .retry_on_initial_connect()
            .ping_interval(Duration::from_secs(10))
            .reconnect_delay_callback(|attempt| {
                Duration::from_millis(std::cmp::min(attempt * 100, 5000) as u64)
            });

        // Add authentication if configured
        if let Some(ref auth) = self.nats_config.auth_config {
            connect_options = self.apply_auth_config(connect_options, auth).await?;
        }

        // Add TLS if configured
        if let Some(ref tls) = self.nats_config.tls_config {
            connect_options = self.apply_tls_config(connect_options, tls)?;
        }

        // Connect with cluster support
        let client = if let Some(ref cluster_urls) = self.nats_config.cluster_urls {
            let all_urls = std::iter::once(self.nats_config.url.clone())
                .chain(cluster_urls.iter().cloned())
                .collect::<Vec<_>>();

            let urls_str = all_urls.join(",");
            async_nats::connect_with_options(urls_str, connect_options)
                .await
                .map_err(|e| anyhow!("Failed to connect to NATS cluster: {}", e))?
        } else {
            async_nats::connect_with_options(self.nats_config.url.clone(), connect_options)
                .await
                .map_err(|e| anyhow!("Failed to connect to NATS: {}", e))?
        };

        let jetstream = jetstream::new(client.clone());

        // Get or create consumer
        let stream = jetstream
            .get_stream(&self.nats_config.stream_name)
            .await
            .map_err(|e| anyhow!("Failed to get JetStream stream: {}", e))?;

        let consumer = stream
            .get_or_create_consumer(
                &self.nats_config.consumer_config.name,
                jetstream::consumer::pull::Config {
                    durable_name: Some(self.nats_config.consumer_config.name.clone()),
                    description: Some(self.nats_config.consumer_config.description.clone()),
                    ..Default::default()
                },
            )
            .await
            .map_err(|e| anyhow!("Failed to get or create consumer: {}", e))?;

        self.client = Some(client);
        self.jetstream = Some(jetstream);
        self.consumer = Some(consumer);

        info!(
            "Connected NATS consumer to {} (cluster mode: {})",
            self.nats_config.url,
            self.nats_config.cluster_urls.is_some()
        );
        Ok(())
    }

    #[cfg(not(feature = "nats"))]
    pub async fn connect(&mut self) -> Result<()> {
        info!("NATS feature not enabled, using mock consumer connection");
        Ok(())
    }

    pub async fn consume(&mut self) -> Result<Option<StreamEvent>> {
        #[cfg(feature = "nats")]
        {
            if self.consumer.is_none() {
                self.connect().await?;
            }

            if let Some(ref mut consumer) = self.consumer {
                // Fetch a single message with timeout
                match tokio::time::timeout(
                    Duration::from_millis(100),
                    consumer.batch().max_messages(1).messages(),
                )
                .await
                {
                    Ok(Ok(mut messages)) => {
                        if let Some(Ok(msg)) = messages.next().await {
                            let payload = msg.payload.clone();
                            let payload_len = payload.len();

                            // Deserialize the message
                            match serde_json::from_slice::<NatsEventMessage>(&payload) {
                                Ok(nats_event) => {
                                    // Acknowledge the message
                                    if let Err(e) = msg.ack().await {
                                        error!("Failed to acknowledge message: {}", e);
                                    }

                                    let mut stats = self.stats.write().await;
                                    stats.events_consumed += 1;
                                    stats.bytes_received += payload_len as u64;
                                    stats.last_consume = Some(chrono::Utc::now());

                                    debug!("Consumed event from NATS: {}", nats_event.event_id);
                                    let stream_event: StreamEvent =
                                        nats_event.try_into().map_err(|e| {
                                            anyhow!("Failed to convert NATS message: {}", e)
                                        })?;
                                    Ok(Some(stream_event))
                                }
                                Err(e) => {
                                    error!("Failed to deserialize NATS message: {}", e);
                                    // Acknowledge to avoid reprocessing bad messages
                                    let _ = msg.ack().await;

                                    let mut stats = self.stats.write().await;
                                    stats.events_failed += 1;

                                    Err(anyhow!("Failed to deserialize message: {}", e))
                                }
                            }
                        } else {
                            // No messages available
                            Ok(None)
                        }
                    }
                    Ok(Err(e)) => {
                        let mut stats = self.stats.write().await;
                        stats.events_failed += 1;
                        Err(anyhow!("Failed to fetch messages: {}", e))
                    }
                    Err(_) => {
                        // Timeout - no messages available
                        Ok(None)
                    }
                }
            } else {
                Err(anyhow!("NATS consumer not initialized"))
            }
        }
        #[cfg(not(feature = "nats"))]
        {
            debug!("Mock NATS consume");
            Ok(None)
        }
    }

    pub async fn get_stats(&self) -> ConsumerStats {
        self.stats.read().await.clone()
    }
}
