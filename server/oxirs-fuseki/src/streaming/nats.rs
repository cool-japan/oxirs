//! NATS integration for lightweight event streaming

use std::sync::Arc;
use async_trait::async_trait;
use async_nats::{Client, jetstream, Subscriber};
use bytes::Bytes;
use futures::StreamExt;
use serde_json;
use tokio::sync::{Mutex, RwLock};

use crate::{
    error::{Error, Result},
    streaming::{NatsConfig, NatsAuth, RDFEvent, StreamProducer, StreamConsumer, EventHandler},
};

/// NATS producer for sending RDF events
pub struct NatsProducer {
    client: Client,
    jetstream: Option<jetstream::Context>,
    config: NatsConfig,
}

impl NatsProducer {
    /// Create a new NATS producer
    pub async fn new(config: NatsConfig) -> Result<Self> {
        // Build connection options
        let mut options = async_nats::ConnectOptions::new();
        
        // Set server URLs
        for url in &config.servers {
            options = options.add_url(url.as_str());
        }
        
        // Configure authentication
        if let Some(auth) = &config.auth {
            options = match auth {
                NatsAuth::UserPass { username, password } => {
                    options.user_and_password(username.clone(), password.clone())
                }
                NatsAuth::Token(token) => {
                    options.token(token.clone())
                }
                NatsAuth::NKey { seed } => {
                    options.nkey(seed.clone())
                }
            };
        }
        
        // Connect to NATS
        let client = options.connect()
            .await
            .map_err(|e| Error::Custom(format!("Failed to connect to NATS: {}", e)))?;
        
        // Create JetStream context if enabled
        let jetstream = if config.jetstream {
            let js = jetstream::new(client.clone());
            
            // Create streams for different event types
            let stream_config = jetstream::stream::Config {
                name: format!("{}_events", config.subject_prefix),
                subjects: vec![format!("{}.*", config.subject_prefix)],
                retention: jetstream::stream::RetentionPolicy::WorkQueue,
                storage: jetstream::stream::StorageType::File,
                ..Default::default()
            };
            
            js.create_stream(stream_config).await
                .map_err(|e| Error::Custom(format!("Failed to create JetStream stream: {}", e)))?;
            
            Some(js)
        } else {
            None
        };
        
        Ok(Self {
            client,
            jetstream,
            config,
        })
    }

    /// Get subject name for an event type
    fn get_subject_name(&self, event: &RDFEvent) -> String {
        let event_type = match event {
            RDFEvent::TripleAdded { .. } => "triple.added",
            RDFEvent::TripleRemoved { .. } => "triple.removed",
            RDFEvent::QuadAdded { .. } => "quad.added",
            RDFEvent::QuadRemoved { .. } => "quad.removed",
            RDFEvent::GraphCleared { .. } => "graph.cleared",
            RDFEvent::Transaction { .. } => "transaction",
        };
        
        format!("{}.{}", self.config.subject_prefix, event_type)
    }
}

#[async_trait]
impl StreamProducer for NatsProducer {
    async fn send(&self, event: RDFEvent) -> Result<()> {
        let subject = self.get_subject_name(&event);
        let payload = serde_json::to_vec(&event)
            .map_err(|e| Error::Custom(format!("Failed to serialize event: {}", e)))?;
        
        if let Some(js) = &self.jetstream {
            // Publish to JetStream
            let ack = js.publish(subject, payload.into()).await
                .map_err(|e| Error::Custom(format!("Failed to publish to JetStream: {}", e)))?
                .await
                .map_err(|e| Error::Custom(format!("Failed to get publish ack: {}", e)))?;
            
            tracing::debug!(
                "Event published to JetStream subject {} with sequence {}",
                ack.stream, ack.sequence
            );
        } else {
            // Regular NATS publish
            self.client.publish(subject, payload.into()).await
                .map_err(|e| Error::Custom(format!("Failed to publish to NATS: {}", e)))?;
            
            tracing::debug!("Event published to NATS subject {}", subject);
        }
        
        Ok(())
    }

    async fn send_batch(&self, events: Vec<RDFEvent>) -> Result<()> {
        // NATS doesn't have built-in transactions, so we send individually
        for event in events {
            self.send(event).await?;
        }
        Ok(())
    }

    async fn flush(&self) -> Result<()> {
        self.client.flush().await
            .map_err(|e| Error::Custom(format!("Failed to flush NATS client: {}", e)))?;
        Ok(())
    }
}

/// NATS consumer for receiving RDF events
pub struct NatsConsumer {
    client: Client,
    jetstream: Option<jetstream::Context>,
    config: NatsConfig,
    handler: Arc<Mutex<Option<Box<dyn EventHandler>>>>,
    subscriptions: Arc<RwLock<Vec<Subscriber>>>,
}

impl NatsConsumer {
    /// Create a new NATS consumer
    pub async fn new(config: NatsConfig) -> Result<Self> {
        // Connect using same logic as producer
        let mut options = async_nats::ConnectOptions::new();
        
        for url in &config.servers {
            options = options.add_url(url.as_str());
        }
        
        if let Some(auth) = &config.auth {
            options = match auth {
                NatsAuth::UserPass { username, password } => {
                    options.user_and_password(username.clone(), password.clone())
                }
                NatsAuth::Token(token) => {
                    options.token(token.clone())
                }
                NatsAuth::NKey { seed } => {
                    options.nkey(seed.clone())
                }
            };
        }
        
        let client = options.connect()
            .await
            .map_err(|e| Error::Custom(format!("Failed to connect to NATS: {}", e)))?;
        
        let jetstream = if config.jetstream {
            Some(jetstream::new(client.clone()))
        } else {
            None
        };
        
        Ok(Self {
            client,
            jetstream,
            config,
            handler: Arc::new(Mutex::new(None)),
            subscriptions: Arc::new(RwLock::new(Vec::new())),
        })
    }

    /// Start consuming messages
    async fn start_consuming(&self) {
        let subjects = vec![
            format!("{}.triple.added", self.config.subject_prefix),
            format!("{}.triple.removed", self.config.subject_prefix),
            format!("{}.quad.added", self.config.subject_prefix),
            format!("{}.quad.removed", self.config.subject_prefix),
            format!("{}.graph.cleared", self.config.subject_prefix),
            format!("{}.transaction", self.config.subject_prefix),
        ];
        
        let mut subs = self.subscriptions.write().await;
        
        for subject in subjects {
            let mut subscriber = if let Some(js) = &self.jetstream {
                // Create durable consumer for JetStream
                let consumer_config = jetstream::consumer::pull::Config {
                    durable_name: Some(format!("oxirs-consumer-{}", subject.replace('.', "-"))),
                    ..Default::default()
                };
                
                let consumer = js.create_consumer_on_stream(
                    consumer_config,
                    format!("{}_events", self.config.subject_prefix),
                ).await.map_err(|e| {
                    Error::Custom(format!("Failed to create JetStream consumer: {}", e))
                })?;
                
                consumer.messages().await.map_err(|e| {
                    Error::Custom(format!("Failed to get message stream: {}", e))
                })?
            } else {
                // Regular NATS subscription
                self.client.subscribe(subject.clone()).await.map_err(|e| {
                    Error::Custom(format!("Failed to subscribe to subject {}: {}", subject, e))
                })?
            };
            
            let handler = self.handler.clone();
            
            // Spawn task to handle messages
            tokio::spawn(async move {
                while let Some(message) = subscriber.next().await {
                    if let Ok(event) = serde_json::from_slice::<RDFEvent>(&message.payload) {
                        if let Some(handler) = &*handler.lock().await {
                            if let Err(e) = handler.handle(event).await {
                                handler.on_error(Box::new(e)).await;
                            }
                        }
                    } else {
                        tracing::error!("Failed to deserialize event from subject {}", message.subject);
                    }
                    
                    // Acknowledge message if using JetStream
                    if message.headers.is_some() {
                        if let Err(e) = message.ack().await {
                            tracing::error!("Failed to acknowledge message: {}", e);
                        }
                    }
                }
            });
            
            subs.push(subscriber);
        }
    }
}

#[async_trait]
impl StreamConsumer for NatsConsumer {
    async fn subscribe(&self, handler: Box<dyn EventHandler>) -> Result<()> {
        *self.handler.lock().await = Some(handler);
        self.start_consuming().await;
        Ok(())
    }

    async fn unsubscribe(&self) -> Result<()> {
        let mut subs = self.subscriptions.write().await;
        subs.clear(); // This will drop all subscriptions
        Ok(())
    }

    async fn commit(&self) -> Result<()> {
        // NATS uses message acknowledgment, not offset commits
        // Messages are acknowledged as they are processed
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use url::Url;

    #[test]
    fn test_subject_naming() {
        let config = NatsConfig {
            servers: vec![Url::parse("nats://localhost:4222").unwrap()],
            subject_prefix: "oxirs".to_string(),
            jetstream: false,
            auth: None,
        };
        
        let producer = NatsProducer {
            client: unsafe { std::mem::zeroed() }, // Don't actually use this
            jetstream: None,
            config,
        };
        
        let event = RDFEvent::TripleAdded {
            triple: unsafe { std::mem::zeroed() }, // Don't actually use this
            graph: Some("test".to_string()),
            timestamp: 12345,
        };
        
        assert_eq!(producer.get_subject_name(&event), "oxirs.triple.added");
    }

    #[test]
    fn test_auth_config() {
        let auth = NatsAuth::UserPass {
            username: "user".to_string(),
            password: "pass".to_string(),
        };
        
        match auth {
            NatsAuth::UserPass { username, password } => {
                assert_eq!(username, "user");
                assert_eq!(password, "pass");
            }
            _ => panic!("Wrong auth type"),
        }
    }
}