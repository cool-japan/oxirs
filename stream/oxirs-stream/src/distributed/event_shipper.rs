//! # Event shipper
//!
//! Cross-shard event delivery for distributed stream processing. Events that
//! arrive at a node but belong to a shard owned by another node are shipped
//! through this layer.
//!
//! ## Transport
//!
//! The shipper uses an in-process channel transport that is interface-compatible
//! with a network transport: a sender posts a [`ShippedEvent`] tagged with its
//! destination node id, and the receiver — running on the destination node —
//! unpacks it. Production deployments swap the in-process implementation with
//! one backed by the existing oxirs-stream backends (e.g. NATS or gRPC).
//!
//! The trait surface is intentionally minimal so that the cross-shard plumbing
//! can be unit-tested without spinning up real network sockets.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::mpsc;
use tracing::debug;

use super::shard_manager::{NodeId, ShardId};

// ─── Errors ─────────────────────────────────────────────────────────────────

/// Errors raised by [`EventShipper`].
#[derive(Debug, Error)]
pub enum ShipperError {
    /// The destination node has no registered receiver.
    #[error("no route to node {0}")]
    NoRoute(NodeId),
    /// Underlying channel send failed (receiver dropped).
    #[error("send failed: {0}")]
    Send(String),
    /// Local routing rejected the message (bad shard).
    #[error("invalid shard {0}")]
    InvalidShard(ShardId),
}

/// Convenience alias.
pub type ShipperResult<T> = std::result::Result<T, ShipperError>;

// ─── ShippedEvent ──────────────────────────────────────────────────────────

/// A keyed event being shipped between shards. The body is opaque so the
/// shipper can carry any operator payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShippedEvent {
    /// Destination shard id.
    pub shard: ShardId,
    /// Routing key (used by the operator to look up state on arrival).
    pub key: String,
    /// Opaque payload (typically JSON-encoded).
    pub payload: Vec<u8>,
    /// Source node id (for tracing).
    pub source: NodeId,
}

impl ShippedEvent {
    /// Build a new event with a JSON payload.
    pub fn json(
        shard: ShardId,
        key: impl Into<String>,
        payload: &serde_json::Value,
        source: impl Into<NodeId>,
    ) -> ShipperResult<Self> {
        let bytes = serde_json::to_vec(payload).map_err(|e| ShipperError::Send(e.to_string()))?;
        Ok(Self {
            shard,
            key: key.into(),
            payload: bytes,
            source: source.into(),
        })
    }

    /// Attempt to decode the payload as JSON.
    pub fn json_payload(&self) -> ShipperResult<serde_json::Value> {
        serde_json::from_slice(&self.payload).map_err(|e| ShipperError::Send(e.to_string()))
    }
}

// ─── Stats ─────────────────────────────────────────────────────────────────

/// Runtime statistics for [`EventShipper`].
#[derive(Debug, Default)]
pub struct ShipperStats {
    pub sent: AtomicU64,
    pub local: AtomicU64,
    pub no_route: AtomicU64,
    pub send_failures: AtomicU64,
}

impl ShipperStats {
    /// Snapshot the counters into a serializable shape.
    pub fn snapshot(&self) -> ShipperStatsSnapshot {
        ShipperStatsSnapshot {
            sent: self.sent.load(Ordering::Relaxed),
            local: self.local.load(Ordering::Relaxed),
            no_route: self.no_route.load(Ordering::Relaxed),
            send_failures: self.send_failures.load(Ordering::Relaxed),
        }
    }
}

/// Plain serialisable snapshot of [`ShipperStats`].
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ShipperStatsSnapshot {
    pub sent: u64,
    pub local: u64,
    pub no_route: u64,
    pub send_failures: u64,
}

// ─── Transport trait ───────────────────────────────────────────────────────

/// Pluggable transport. The default in-process implementation is below.
#[async_trait]
pub trait ShipperTransport: Send + Sync {
    /// Send an event to the destination node. The transport should not assume
    /// any ordering guarantee across calls — operators are responsible for
    /// reordering events as needed.
    async fn send(&self, dst: &NodeId, event: ShippedEvent) -> ShipperResult<()>;

    /// Optional: register a node so [`ShipperTransport::send`] can route to it.
    /// The default no-op implementation suits transports that discover nodes
    /// out-of-band.
    fn register_route(&self, _node: NodeId) {}

    /// Whether the transport currently knows how to reach `node`.
    fn has_route(&self, _node: &NodeId) -> bool {
        true
    }
}

// ─── In-process transport ──────────────────────────────────────────────────

/// In-process shipping transport. Backed by a `tokio::sync::mpsc` channel per
/// destination node.
pub struct InProcessShipperTransport {
    senders: RwLock<HashMap<NodeId, mpsc::Sender<ShippedEvent>>>,
    capacity: usize,
}

impl InProcessShipperTransport {
    /// Create a new transport with the configured per-node channel capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            senders: RwLock::new(HashMap::new()),
            capacity: capacity.max(1),
        }
    }

    /// Spawn a receiver on the given node. Returns the receiver half so a
    /// caller can drive a consumer loop.
    pub fn spawn_receiver(&self, node: NodeId) -> mpsc::Receiver<ShippedEvent> {
        let (tx, rx) = mpsc::channel(self.capacity);
        self.senders.write().insert(node, tx);
        rx
    }
}

#[async_trait]
impl ShipperTransport for InProcessShipperTransport {
    async fn send(&self, dst: &NodeId, event: ShippedEvent) -> ShipperResult<()> {
        let sender_opt = self.senders.read().get(dst).cloned();
        match sender_opt {
            Some(tx) => tx
                .send(event)
                .await
                .map_err(|e| ShipperError::Send(e.to_string())),
            None => Err(ShipperError::NoRoute(dst.clone())),
        }
    }

    fn register_route(&self, node: NodeId) {
        // No-op unless `spawn_receiver` is called explicitly. We expose it so
        // the trait surface is complete.
        let _ = node;
    }

    fn has_route(&self, node: &NodeId) -> bool {
        self.senders.read().contains_key(node)
    }
}

// ─── EventShipper ───────────────────────────────────────────────────────────

/// Configuration for [`EventShipper`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShipperConfig {
    /// Identifier of this node — the shipper short-circuits when the
    /// destination matches.
    pub local_node: NodeId,
}

/// High-level cross-shard event router. Delegates to a [`ShipperTransport`] for
/// inter-node delivery and to an in-process channel for events whose
/// destination is the local node.
pub struct EventShipper {
    config: ShipperConfig,
    transport: Arc<dyn ShipperTransport>,
    local_in: parking_lot::Mutex<Option<mpsc::Sender<ShippedEvent>>>,
    stats: Arc<ShipperStats>,
}

impl EventShipper {
    /// Build a shipper.
    pub fn new(config: ShipperConfig, transport: Arc<dyn ShipperTransport>) -> Self {
        Self {
            config,
            transport,
            local_in: parking_lot::Mutex::new(None),
            stats: Arc::new(ShipperStats::default()),
        }
    }

    /// Install a local-delivery sink so events targeting the local node are
    /// dispatched through it (the `Receiver` half is held by the operator).
    pub fn install_local_sink(&self, sink: mpsc::Sender<ShippedEvent>) {
        *self.local_in.lock() = Some(sink);
    }

    /// Stats accessor.
    pub fn stats(&self) -> &Arc<ShipperStats> {
        &self.stats
    }

    /// Local node id.
    pub fn local_node(&self) -> &NodeId {
        &self.config.local_node
    }

    /// Send a single event to its destination node. If the destination is the
    /// local node, the local sink is used; otherwise the transport ships it.
    pub async fn ship(&self, dst: &NodeId, event: ShippedEvent) -> ShipperResult<()> {
        if dst == &self.config.local_node {
            let sink_opt = self.local_in.lock().as_ref().cloned();
            return match sink_opt {
                Some(tx) => {
                    tx.send(event).await.map_err(|e| {
                        self.stats.send_failures.fetch_add(1, Ordering::Relaxed);
                        ShipperError::Send(e.to_string())
                    })?;
                    self.stats.local.fetch_add(1, Ordering::Relaxed);
                    Ok(())
                }
                None => {
                    self.stats.no_route.fetch_add(1, Ordering::Relaxed);
                    Err(ShipperError::NoRoute(dst.clone()))
                }
            };
        }
        if !self.transport.has_route(dst) {
            self.stats.no_route.fetch_add(1, Ordering::Relaxed);
            return Err(ShipperError::NoRoute(dst.clone()));
        }
        match self.transport.send(dst, event).await {
            Ok(()) => {
                self.stats.sent.fetch_add(1, Ordering::Relaxed);
                debug!(dst = %dst, "ship: ok");
                Ok(())
            }
            Err(err) => {
                self.stats.send_failures.fetch_add(1, Ordering::Relaxed);
                Err(err)
            }
        }
    }

    /// Ship a batch of events. The shipper iterates and forwards each event
    /// individually; the in-process transport preserves order per destination.
    pub async fn ship_batch(&self, dst: &NodeId, events: Vec<ShippedEvent>) -> ShipperResult<()> {
        for ev in events {
            self.ship(dst, ev).await?;
        }
        Ok(())
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_event(shard: ShardId) -> ShippedEvent {
        ShippedEvent::json(shard, "k", &serde_json::json!({"v": 1}), "src").expect("ok")
    }

    #[tokio::test]
    async fn local_delivery_uses_installed_sink() {
        let transport = Arc::new(InProcessShipperTransport::new(10));
        let shipper = EventShipper::new(
            ShipperConfig {
                local_node: "self".into(),
            },
            transport,
        );
        let (tx, mut rx) = mpsc::channel(10);
        shipper.install_local_sink(tx);
        shipper
            .ship(&"self".to_string(), make_event(0))
            .await
            .expect("ship");
        let received = rx.recv().await.expect("received");
        assert_eq!(received.shard, 0);
        let stats = shipper.stats().snapshot();
        assert_eq!(stats.local, 1);
        assert_eq!(stats.sent, 0);
    }

    #[tokio::test]
    async fn remote_delivery_ships_through_transport() {
        let transport = Arc::new(InProcessShipperTransport::new(10));
        let mut rx_remote = transport.spawn_receiver("remote".into());
        let shipper = EventShipper::new(
            ShipperConfig {
                local_node: "self".into(),
            },
            transport,
        );

        shipper
            .ship(&"remote".to_string(), make_event(1))
            .await
            .expect("ship");
        let received = rx_remote.recv().await.expect("received");
        assert_eq!(received.shard, 1);
        let stats = shipper.stats().snapshot();
        assert_eq!(stats.sent, 1);
    }

    #[tokio::test]
    async fn unknown_destination_returns_no_route() {
        let transport = Arc::new(InProcessShipperTransport::new(10));
        let shipper = EventShipper::new(
            ShipperConfig {
                local_node: "self".into(),
            },
            transport,
        );
        let err = shipper
            .ship(&"missing".to_string(), make_event(2))
            .await
            .expect_err("should fail");
        assert!(matches!(err, ShipperError::NoRoute(_)));
    }

    #[tokio::test]
    async fn ship_batch_preserves_order() {
        let transport = Arc::new(InProcessShipperTransport::new(10));
        let mut rx = transport.spawn_receiver("dst".into());
        let shipper = EventShipper::new(
            ShipperConfig {
                local_node: "src".into(),
            },
            transport,
        );
        let events = (0..5).map(|i| make_event(i as u32)).collect::<Vec<_>>();
        shipper
            .ship_batch(&"dst".to_string(), events)
            .await
            .expect("ship");
        let mut received = Vec::new();
        for _ in 0..5 {
            received.push(rx.recv().await.expect("received").shard);
        }
        assert_eq!(received, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn json_payload_round_trips() {
        let payload = serde_json::json!({"hello": "world"});
        let event = ShippedEvent::json(0, "k", &payload, "src").expect("ok");
        let back = event.json_payload().expect("decode");
        assert_eq!(back, payload);
    }
}
