//! Change Data Capture (CDC) for RDF stores

use std::{
    collections::HashMap,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};
use tokio::sync::{mpsc, RwLock};

use crate::{
    error::Result,
    store::Store,
    streaming::{CDCConfig, RDFEvent, StreamingManager},
};
use oxirs_core::{Dataset, Quad, Triple};

/// CDC listener trait for intercepting store operations
#[async_trait::async_trait]
pub trait CDCListener: Send + Sync {
    /// Called when a triple is added
    async fn on_triple_added(&self, triple: &Triple, graph: Option<&str>);

    /// Called when a triple is removed
    async fn on_triple_removed(&self, triple: &Triple, graph: Option<&str>);

    /// Called when a quad is added
    async fn on_quad_added(&self, quad: &Quad);

    /// Called when a quad is removed
    async fn on_quad_removed(&self, quad: &Quad);

    /// Called when a graph is cleared
    async fn on_graph_cleared(&self, graph: &str);

    /// Called at transaction boundaries
    async fn on_transaction_start(&self, tx_id: &str);
    async fn on_transaction_commit(&self, tx_id: &str);
    async fn on_transaction_rollback(&self, tx_id: &str);
}

/// CDC manager for capturing and streaming data changes
pub struct CDCManager {
    config: CDCConfig,
    streaming_manager: Arc<StreamingManager>,
    transaction_buffer: Arc<RwLock<HashMap<String, Vec<RDFEvent>>>>,
    event_channel: (
        mpsc::Sender<CDCEvent>,
        Arc<RwLock<mpsc::Receiver<CDCEvent>>>,
    ),
}

/// Internal CDC event type
#[derive(Debug, Clone)]
enum CDCEvent {
    TripleAdded {
        triple: Triple,
        graph: Option<String>,
    },
    TripleRemoved {
        triple: Triple,
        graph: Option<String>,
    },
    QuadAdded {
        quad: Quad,
    },
    QuadRemoved {
        quad: Quad,
    },
    GraphCleared {
        graph: String,
    },
    TransactionStart {
        tx_id: String,
    },
    TransactionCommit {
        tx_id: String,
    },
    TransactionRollback {
        tx_id: String,
    },
}

impl CDCManager {
    /// Create a new CDC manager
    pub fn new(config: CDCConfig, streaming_manager: Arc<StreamingManager>) -> Self {
        let (tx, rx) = mpsc::channel(10000);

        Self {
            config,
            streaming_manager,
            transaction_buffer: Arc::new(RwLock::new(HashMap::new())),
            event_channel: (tx, Arc::new(RwLock::new(rx))),
        }
    }

    /// Start the CDC processing loop
    pub async fn start(&self) {
        let config = self.config.clone();
        let streaming_manager = self.streaming_manager.clone();
        let transaction_buffer = self.transaction_buffer.clone();
        let receiver = self.event_channel.1.clone();

        tokio::spawn(async move {
            let mut batch = Vec::new();
            let mut interval = tokio::time::interval(tokio::time::Duration::from_millis(100));

            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        if !batch.is_empty() && batch.len() >= config.batch_size {
                            Self::process_batch(&streaming_manager, &batch).await;
                            batch.clear();
                        }
                    }
                    event = async {
                        let mut rx = receiver.write().await;
                        rx.recv().await
                    } => {
                        if let Some(event) = event {
                            match event {
                                CDCEvent::TransactionStart { tx_id } => {
                                    // Start buffering events for this transaction
                                    let mut buffer = transaction_buffer.write().await;
                                    buffer.insert(tx_id, Vec::new());
                                }
                                CDCEvent::TransactionCommit { tx_id } => {
                                    // Send all buffered events as a transaction
                                    let mut buffer = transaction_buffer.write().await;
                                    if let Some(events) = buffer.remove(&tx_id) {
                                        let timestamp = Self::current_timestamp();
                                        let tx_event = RDFEvent::Transaction {
                                            id: tx_id,
                                            events,
                                            timestamp,
                                        };
                                        let _ = streaming_manager.send_event(tx_event).await;
                                    }
                                }
                                CDCEvent::TransactionRollback { tx_id } => {
                                    // Discard buffered events
                                    let mut buffer = transaction_buffer.write().await;
                                    buffer.remove(&tx_id);
                                }
                                _ => {
                                    // Convert to RDF event and add to batch
                                    if let Some(rdf_event) = Self::convert_to_rdf_event(&event, &config) {
                                        // Check if we're in a transaction
                                        let buffer = transaction_buffer.read().await;
                                        let in_transaction = buffer.keys().next().cloned();
                                        drop(buffer);

                                        if let Some(tx_id) = in_transaction {
                                            let mut buffer = transaction_buffer.write().await;
                                            if let Some(tx_events) = buffer.get_mut(&tx_id) {
                                                tx_events.push(rdf_event);
                                            }
                                        } else {
                                            batch.push(rdf_event);
                                        }
                                    }
                                }
                            }
                        } else {
                            // Channel closed, send remaining batch
                            if !batch.is_empty() {
                                Self::process_batch(&streaming_manager, &batch).await;
                            }
                            break;
                        }
                    }
                }
            }
        });
    }

    /// Process a batch of events
    async fn process_batch(streaming_manager: &StreamingManager, batch: &[RDFEvent]) {
        for event in batch {
            if let Err(e) = streaming_manager.send_event(event.clone()).await {
                tracing::error!("Failed to send CDC event: {}", e);
            }
        }
    }

    /// Convert internal CDC event to RDF event
    fn convert_to_rdf_event(event: &CDCEvent, config: &CDCConfig) -> Option<RDFEvent> {
        let timestamp = Self::current_timestamp();

        match event {
            CDCEvent::TripleAdded { triple, graph } if config.capture_inserts => {
                Some(RDFEvent::TripleAdded {
                    triple: triple.clone(),
                    graph: graph.clone(),
                    timestamp,
                })
            }
            CDCEvent::TripleRemoved { triple, graph } if config.capture_deletes => {
                Some(RDFEvent::TripleRemoved {
                    triple: triple.clone(),
                    graph: graph.clone(),
                    timestamp,
                })
            }
            CDCEvent::QuadAdded { quad } if config.capture_inserts => Some(RDFEvent::QuadAdded {
                quad: quad.clone(),
                timestamp,
            }),
            CDCEvent::QuadRemoved { quad } if config.capture_deletes => {
                Some(RDFEvent::QuadRemoved {
                    quad: quad.clone(),
                    timestamp,
                })
            }
            CDCEvent::GraphCleared { graph } if config.capture_deletes => {
                Some(RDFEvent::GraphCleared {
                    graph: graph.clone(),
                    timestamp,
                })
            }
            _ => None,
        }
    }

    /// Get current timestamp in milliseconds
    fn current_timestamp() -> i64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64
    }
}

#[async_trait::async_trait]
impl CDCListener for CDCManager {
    async fn on_triple_added(&self, triple: &Triple, graph: Option<&str>) {
        if self.config.enabled && self.config.capture_inserts {
            let event = CDCEvent::TripleAdded {
                triple: triple.clone(),
                graph: graph.map(|s| s.to_string()),
            };
            let _ = self.event_channel.0.send(event).await;
        }
    }

    async fn on_triple_removed(&self, triple: &Triple, graph: Option<&str>) {
        if self.config.enabled && self.config.capture_deletes {
            let event = CDCEvent::TripleRemoved {
                triple: triple.clone(),
                graph: graph.map(|s| s.to_string()),
            };
            let _ = self.event_channel.0.send(event).await;
        }
    }

    async fn on_quad_added(&self, quad: &Quad) {
        if self.config.enabled && self.config.capture_inserts {
            let event = CDCEvent::QuadAdded { quad: quad.clone() };
            let _ = self.event_channel.0.send(event).await;
        }
    }

    async fn on_quad_removed(&self, quad: &Quad) {
        if self.config.enabled && self.config.capture_deletes {
            let event = CDCEvent::QuadRemoved { quad: quad.clone() };
            let _ = self.event_channel.0.send(event).await;
        }
    }

    async fn on_graph_cleared(&self, graph: &str) {
        if self.config.enabled && self.config.capture_deletes {
            let event = CDCEvent::GraphCleared {
                graph: graph.to_string(),
            };
            let _ = self.event_channel.0.send(event).await;
        }
    }

    async fn on_transaction_start(&self, tx_id: &str) {
        if self.config.enabled {
            let event = CDCEvent::TransactionStart {
                tx_id: tx_id.to_string(),
            };
            let _ = self.event_channel.0.send(event).await;
        }
    }

    async fn on_transaction_commit(&self, tx_id: &str) {
        if self.config.enabled {
            let event = CDCEvent::TransactionCommit {
                tx_id: tx_id.to_string(),
            };
            let _ = self.event_channel.0.send(event).await;
        }
    }

    async fn on_transaction_rollback(&self, tx_id: &str) {
        if self.config.enabled {
            let event = CDCEvent::TransactionRollback {
                tx_id: tx_id.to_string(),
            };
            let _ = self.event_channel.0.send(event).await;
        }
    }
}

/// Store wrapper that adds CDC capabilities
pub struct CDCStore {
    inner: Store,
    cdc_listeners: Arc<RwLock<Vec<Arc<dyn CDCListener>>>>,
}

impl CDCStore {
    /// Create a new CDC-enabled store
    pub fn new(store: Store) -> Self {
        Self {
            inner: store,
            cdc_listeners: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Add a CDC listener
    pub async fn add_listener(&self, listener: Arc<dyn CDCListener>) {
        let mut listeners = self.cdc_listeners.write().await;
        listeners.push(listener);
    }

    /// Remove all CDC listeners
    pub async fn clear_listeners(&self) {
        let mut listeners = self.cdc_listeners.write().await;
        listeners.clear();
    }

    /// Notify all listeners of an event
    async fn notify_listeners<F, Fut>(&self, f: F)
    where
        F: Fn(Arc<dyn CDCListener>) -> Fut,
        Fut: std::future::Future<Output = ()>,
    {
        let listeners = self.cdc_listeners.read().await;
        for listener in listeners.iter() {
            f(listener.clone()).await;
        }
    }

    // Wrapper methods that trigger CDC events would go here
    // For example:

    pub async fn add_triple(&self, triple: Triple, graph: Option<String>) -> Result<()> {
        // Add to store
        // self.inner.add_triple(triple.clone(), graph.clone()).await?;

        // Notify CDC listeners
        let graph_ref = graph.as_deref();
        self.notify_listeners(|listener| async move {
            listener.on_triple_added(&triple, graph_ref).await;
        })
        .await;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_cdc_config() {
        let config = CDCConfig::default();
        assert!(config.enabled);
        assert!(config.capture_inserts);
        assert!(config.capture_deletes);
        assert!(config.capture_updates);
        assert_eq!(config.batch_size, 100);
    }

    #[test]
    fn test_timestamp_generation() {
        let ts1 = CDCManager::current_timestamp();
        std::thread::sleep(std::time::Duration::from_millis(10));
        let ts2 = CDCManager::current_timestamp();
        assert!(ts2 > ts1);
    }
}
