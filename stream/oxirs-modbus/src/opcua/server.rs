//! OPC UA server facade and mock implementation.
//!
//! The [`OpcuaServerFacade`] trait isolates the bridge from any concrete OPC UA
//! library version so that tests can use the in-process [`MockOpcuaServer`]
//! without opening real network sockets.

use async_trait::async_trait;
use std::sync::{Arc, Mutex};
use thiserror::Error;
use tokio::sync::mpsc;

use crate::opcua::type_coercion::DataValue;

/// Errors that can occur in the OPC UA server facade.
#[derive(Debug, Error)]
pub enum OpcuaError {
    /// The OPC UA server rejected the publish operation.
    #[error("publish failed for node {node_id}: {reason}")]
    PublishFailed { node_id: String, reason: String },

    /// The channel for write-subscriptions is not available (already consumed or never created).
    #[error("write channel not available")]
    WriteChannelUnavailable,

    /// Shutdown encountered an error.
    #[error("shutdown error: {0}")]
    Shutdown(String),

    /// Generic catch-all.
    #[error("OPC UA error: {0}")]
    Other(String),
}

/// Thin abstraction over an OPC UA server, used by [`OpcuaModbusBridge`].
///
/// Implementations must be `Send + Sync` so that the bridge can share them
/// across tokio tasks.
///
/// [`OpcuaModbusBridge`]: crate::opcua::bridge::OpcuaModbusBridge
#[async_trait]
pub trait OpcuaServerFacade: Send + Sync {
    /// Publish a typed value to the OPC UA address space at `node_id`.
    async fn publish_value(&self, node_id: &str, value: &DataValue) -> Result<(), OpcuaError>;

    /// Subscribe to writes coming from OPC UA clients.
    ///
    /// Returns a channel that yields `(node_id, value)` pairs whenever an OPC
    /// UA client writes to a variable managed by this server.  This method
    /// should be called at most once; calling it again after the first
    /// successful call returns [`OpcuaError::WriteChannelUnavailable`].
    async fn subscribe_writes(&mut self)
        -> Result<mpsc::Receiver<(String, DataValue)>, OpcuaError>;

    /// Gracefully shut down the OPC UA server.
    async fn shutdown(&self) -> Result<(), OpcuaError>;
}

/// In-process mock OPC UA server for use in tests.
///
/// Published values are recorded in `published` and write events can be
/// injected via [`MockOpcuaServer::simulate_write`].
pub struct MockOpcuaServer {
    /// All `(node_id, value)` pairs published via [`OpcuaServerFacade::publish_value`].
    pub published: Arc<Mutex<Vec<(String, DataValue)>>>,
    /// Sender half of the write event channel. Use [`simulate_write`] to inject events.
    pub write_tx: mpsc::Sender<(String, DataValue)>,
    /// Receiver half — taken by [`subscribe_writes`].
    write_rx: Option<mpsc::Receiver<(String, DataValue)>>,
}

impl Default for MockOpcuaServer {
    fn default() -> Self {
        Self::new()
    }
}

impl MockOpcuaServer {
    /// Create a new mock server with an unbounded-ish channel (capacity 64).
    pub fn new() -> Self {
        let (write_tx, write_rx) = mpsc::channel(64);
        Self {
            published: Arc::new(Mutex::new(Vec::new())),
            write_tx,
            write_rx: Some(write_rx),
        }
    }

    /// Inject a simulated OPC UA client write event.
    ///
    /// The bridge's subscription loop will see this as if a real OPC UA client
    /// had written `value` to `node_id`.
    pub fn simulate_write(&self, node_id: &str, value: DataValue) {
        // Best-effort; ignore if channel is closed.
        let _ = self.write_tx.try_send((node_id.to_owned(), value));
    }

    /// Return a clone of the published log for assertions.
    pub fn get_published(&self) -> Vec<(String, DataValue)> {
        self.published
            .lock()
            .expect("published lock poisoned")
            .clone()
    }
}

#[async_trait]
impl OpcuaServerFacade for MockOpcuaServer {
    async fn publish_value(&self, node_id: &str, value: &DataValue) -> Result<(), OpcuaError> {
        self.published
            .lock()
            .map_err(|_| OpcuaError::Other("lock poisoned".to_owned()))?
            .push((node_id.to_owned(), value.clone()));
        Ok(())
    }

    async fn subscribe_writes(
        &mut self,
    ) -> Result<mpsc::Receiver<(String, DataValue)>, OpcuaError> {
        self.write_rx
            .take()
            .ok_or(OpcuaError::WriteChannelUnavailable)
    }

    async fn shutdown(&self) -> Result<(), OpcuaError> {
        // Nothing to do for the mock.
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn mock_publish_and_retrieve() {
        let server = MockOpcuaServer::new();
        server
            .publish_value("ns=2;s=Temp", &DataValue::F32(23.5))
            .await
            .expect("publish ok");
        let log = server.get_published();
        assert_eq!(log.len(), 1);
        assert_eq!(log[0].0, "ns=2;s=Temp");
        assert_eq!(log[0].1, DataValue::F32(23.5));
    }

    #[tokio::test]
    async fn mock_subscribe_writes() {
        let mut server = MockOpcuaServer::new();
        server.simulate_write("ns=2;s=Pump", DataValue::Bool(true));
        let mut rx = server.subscribe_writes().await.expect("subscribe ok");
        let (node, val) = rx.recv().await.expect("should receive");
        assert_eq!(node, "ns=2;s=Pump");
        assert_eq!(val, DataValue::Bool(true));
    }

    #[tokio::test]
    async fn mock_subscribe_writes_unavailable_on_second_call() {
        let mut server = MockOpcuaServer::new();
        server.subscribe_writes().await.expect("first ok");
        let err = server.subscribe_writes().await.unwrap_err();
        assert!(matches!(err, OpcuaError::WriteChannelUnavailable));
    }
}
