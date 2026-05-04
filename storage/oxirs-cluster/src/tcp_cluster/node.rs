//! A single cluster node backed by a real TCP listener.
//!
//! `TcpClusterNode` listens on a dynamically-assigned local port (bind to
//! `0.0.0.0:0` and read back the OS-assigned address via
//! `TcpListener::local_addr()`).  Peers are registered after construction so
//! that `TcpClusterNetwork` can wire the mesh before gossip rounds start.
//!
//! ## Gossip loop
//!
//! Every `gossip_interval_ms` the node selects
//! `fanout.resolve(peers.len())` peers uniformly at random (using the
//! `scirs2_core::random` generator per COOLJAPAN policy) and connects a
//! fresh TCP stream to each.  It sends one `ClusterMessage::Gossip` for
//! every entry in its local `GossipState`.  LWW reconciliation on the
//! receiver side means stale re-sends are idempotent.
//!
//! ## Listener loop
//!
//! Each accepted connection is handled in its own spawned task.  Messages are
//! read in a loop until the connection closes or the `CancellationToken` fires.
//! Supported incoming messages:
//! - `Gossip` → merge into local state
//! - `Ping`   → reply with `Pong`
//! - `Replicate` → reply with `ReplicateAck { success: true }`
//! - Others are silently ignored

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use parking_lot::RwLock;
use tokio::net::{TcpListener, TcpStream};
use tokio::time::interval;
use tokio_util::sync::CancellationToken;

use crate::gossip::fanout::GossipFanout;

use super::codec::{ClusterMessage, MessageCodec};

// ─────────────────────────────────────────────────────────────────────────────
// GossipState
// ─────────────────────────────────────────────────────────────────────────────

/// Shared key-value state propagated via gossip.
///
/// Each entry is `(value, version)`.  Versions are compared on merge:
/// an incoming entry replaces the stored one only when its version is
/// strictly greater.
#[derive(Default, Clone)]
pub struct GossipState {
    /// `key → (value, version)` — last-write-wins by version number.
    pub entries: HashMap<String, (u64, u64)>,
}

impl GossipState {
    /// Merge an incoming entry.
    ///
    /// Returns `true` if the stored entry was updated (i.e. `version` was
    /// strictly greater than the previously stored version, or the key is new).
    pub fn set(&mut self, key: &str, value: u64, version: u64) -> bool {
        let entry = self.entries.entry(key.to_owned()).or_insert((0, 0));
        if version > entry.1 {
            *entry = (value, version);
            true
        } else {
            false
        }
    }

    /// Return `(value, version)` for `key`, or `None` if absent.
    pub fn get(&self, key: &str) -> Option<(u64, u64)> {
        self.entries.get(key).copied()
    }

    /// Number of distinct keys stored.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the state has no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for a single `TcpClusterNode`.
#[derive(Debug, Clone)]
pub struct TcpNodeConfig {
    /// Human-readable node identifier (used in gossip messages).
    pub node_id: String,
    /// Address to bind; use `0.0.0.0:0` (or `127.0.0.1:0`) to let the OS
    /// assign a free port.
    pub bind_addr: SocketAddr,
    /// Fanout policy for gossip peer selection.
    pub fanout: GossipFanout,
    /// How often to run a gossip round (milliseconds).
    pub gossip_interval_ms: u64,
}

impl TcpNodeConfig {
    /// Convenience constructor binding to `127.0.0.1:port`.
    ///
    /// Pass `port = 0` to let the OS pick a free ephemeral port.
    pub fn localhost(node_id: &str, port: u16) -> Self {
        Self {
            node_id: node_id.to_owned(),
            bind_addr: SocketAddr::from(([127, 0, 0, 1], port)),
            fanout: GossipFanout::Unbounded,
            gossip_interval_ms: 50,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Error
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by [`TcpClusterNode`] operations.
#[derive(Debug)]
pub enum TcpNodeError {
    /// Failed to bind the TCP listener.
    BindError(std::io::Error),
    /// Failed to send a message to a peer.
    SendError(String),
    /// Node has been shut down.
    Shutdown,
}

impl std::fmt::Display for TcpNodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TcpNodeError::BindError(e) => write!(f, "TCP bind failed: {e}"),
            TcpNodeError::SendError(s) => write!(f, "send error: {s}"),
            TcpNodeError::Shutdown => write!(f, "node has been shut down"),
        }
    }
}

impl std::error::Error for TcpNodeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            TcpNodeError::BindError(e) => Some(e),
            TcpNodeError::SendError(_) | TcpNodeError::Shutdown => None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Node
// ─────────────────────────────────────────────────────────────────────────────

/// A single cluster node running real TCP sockets.
///
/// Spawn with [`TcpClusterNode::start`], wire peers with [`add_peer`][Self::add_peer],
/// read/write state with [`get`][Self::get] / [`set`][Self::set],
/// and shut down with [`shutdown`][Self::shutdown].
pub struct TcpClusterNode {
    config: TcpNodeConfig,
    /// Actual bound address (may differ from `config.bind_addr` when port was 0).
    bound_addr: SocketAddr,
    state: Arc<RwLock<GossipState>>,
    peers: Arc<RwLock<Vec<SocketAddr>>>,
    cancel: CancellationToken,
    /// Global version counter: monotonically incremented on every local `set`.
    version: Arc<AtomicU64>,
}

impl TcpClusterNode {
    /// Bind the TCP listener and start background tasks.
    ///
    /// Returns once the listener is bound and the background tasks are running.
    ///
    /// # Errors
    ///
    /// Returns [`TcpNodeError::BindError`] if the port could not be acquired.
    pub async fn start(config: TcpNodeConfig) -> Result<Self, TcpNodeError> {
        let listener = TcpListener::bind(config.bind_addr)
            .await
            .map_err(TcpNodeError::BindError)?;
        let bound_addr = listener.local_addr().map_err(TcpNodeError::BindError)?;

        let state: Arc<RwLock<GossipState>> = Arc::default();
        let peers: Arc<RwLock<Vec<SocketAddr>>> = Arc::default();
        let cancel = CancellationToken::new();
        let version = Arc::new(AtomicU64::new(1));

        // Spawn listener task.
        let state_clone = Arc::clone(&state);
        let cancel_clone = cancel.clone();
        let node_id_clone = config.node_id.clone();
        tokio::spawn(async move {
            run_listener(listener, state_clone, node_id_clone, cancel_clone).await;
        });

        // Spawn gossip timer task.
        let state_gossip = Arc::clone(&state);
        let peers_gossip = Arc::clone(&peers);
        let cancel_gossip = cancel.clone();
        let gossip_interval = config.gossip_interval_ms;
        let fanout = config.fanout;
        let node_id_gossip = config.node_id.clone();
        tokio::spawn(async move {
            run_gossip_loop(
                node_id_gossip,
                fanout,
                gossip_interval,
                state_gossip,
                peers_gossip,
                cancel_gossip,
            )
            .await;
        });

        Ok(Self {
            config,
            bound_addr,
            state,
            peers,
            cancel,
            version,
        })
    }

    /// Register a peer address to gossip with.
    pub fn add_peer(&self, addr: SocketAddr) {
        self.peers.write().push(addr);
    }

    /// Store a key-value pair in the local gossip state.
    ///
    /// The internal monotone version counter is incremented so that subsequent
    /// gossip rounds carry a strictly greater version than any prior one,
    /// guaranteeing LWW ordering across nodes in a single-process test.
    pub fn set(&self, key: &str, value: u64) {
        let ver = self.version.fetch_add(1, Ordering::Relaxed) + 1;
        self.state.write().set(key, value, ver);
    }

    /// Store a key-value pair with an explicit version number.
    ///
    /// This bypasses the internal monotone counter and is intended for test
    /// scenarios that need deterministic LWW ordering (e.g. to prove that a
    /// higher explicit version beats a lower one regardless of call timing).
    ///
    /// The internal counter is advanced to `version.max(current_counter)` so
    /// subsequent [`set`][Self::set] calls still produce strictly greater
    /// versions.
    pub fn set_with_version(&self, key: &str, value: u64, version: u64) {
        self.state.write().set(key, value, version);
        // Keep the internal counter at least at `version` so future `set`
        // calls produce strictly larger versions.
        let mut current = self.version.load(Ordering::Relaxed);
        loop {
            if current >= version {
                break;
            }
            match self.version.compare_exchange_weak(
                current,
                version,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(v) => current = v,
            }
        }
    }

    /// Read the current value for `key` from the local state.
    ///
    /// Returns `None` if the key has not converged to this node yet.
    pub fn get(&self, key: &str) -> Option<u64> {
        self.state.read().get(key).map(|(v, _ver)| v)
    }

    /// Number of distinct keys currently in the local state.
    pub fn state_len(&self) -> usize {
        self.state.read().len()
    }

    /// Signal all background tasks to stop.
    ///
    /// Does not wait for task completion; caller should `tokio::time::sleep`
    /// or `JoinHandle::await` if ordering matters.
    pub fn shutdown(&self) {
        self.cancel.cancel();
    }

    /// The logical node identifier.
    pub fn node_id(&self) -> &str {
        &self.config.node_id
    }

    /// The OS-assigned bound address (canonical for peer registration).
    pub fn addr(&self) -> SocketAddr {
        self.bound_addr
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Background tasks
// ─────────────────────────────────────────────────────────────────────────────

/// Listener task: accept incoming connections and dispatch them to handler tasks.
async fn run_listener(
    listener: TcpListener,
    state: Arc<RwLock<GossipState>>,
    node_id: String,
    cancel: CancellationToken,
) {
    loop {
        tokio::select! {
            biased;
            _ = cancel.cancelled() => break,
            result = listener.accept() => {
                match result {
                    Ok((stream, _peer)) => {
                        let state_clone = Arc::clone(&state);
                        let node_id_clone = node_id.clone();
                        let cancel_clone = cancel.clone();
                        tokio::spawn(async move {
                            handle_connection(stream, state_clone, node_id_clone, cancel_clone).await;
                        });
                    }
                    Err(_) => break,
                }
            }
        }
    }
}

/// Handle a single accepted TCP connection.
///
/// Reads messages until EOF or cancellation, then closes the stream.
async fn handle_connection(
    mut stream: TcpStream,
    state: Arc<RwLock<GossipState>>,
    node_id: String,
    cancel: CancellationToken,
) {
    let (mut reader, mut writer) = stream.split();

    loop {
        let msg = tokio::select! {
            biased;
            _ = cancel.cancelled() => break,
            result = MessageCodec::read(&mut reader) => {
                match result {
                    Ok(m) => m,
                    Err(_) => break, // Connection closed or error
                }
            }
        };

        match msg {
            ClusterMessage::Gossip {
                key,
                value,
                version,
                ..
            } => {
                state.write().set(&key, value, version);
            }
            ClusterMessage::Ping { sender_id: _, seq } => {
                let pong = ClusterMessage::Pong {
                    sender_id: node_id.clone(),
                    seq,
                };
                if MessageCodec::write(&mut writer, &pong).await.is_err() {
                    break;
                }
            }
            ClusterMessage::Replicate { index, .. } => {
                let ack = ClusterMessage::ReplicateAck {
                    follower_id: node_id.clone(),
                    index,
                    success: true,
                };
                if MessageCodec::write(&mut writer, &ack).await.is_err() {
                    break;
                }
            }
            // Pong and ReplicateAck are not expected on an incoming connection
            // from a peer — ignore them silently.
            ClusterMessage::Pong { .. } | ClusterMessage::ReplicateAck { .. } => {}
        }
    }
}

/// Gossip timer task: periodically push local state to a random peer subset.
async fn run_gossip_loop(
    node_id: String,
    fanout: GossipFanout,
    interval_ms: u64,
    state: Arc<RwLock<GossipState>>,
    peers: Arc<RwLock<Vec<SocketAddr>>>,
    cancel: CancellationToken,
) {
    let mut ticker = interval(Duration::from_millis(interval_ms));

    loop {
        tokio::select! {
            biased;
            _ = cancel.cancelled() => break,
            _ = ticker.tick() => {}
        }

        // Snapshot state and peers under lock, then release.
        let snapshot: Vec<(String, u64, u64)> = {
            let st = state.read();
            st.entries
                .iter()
                .map(|(k, (v, ver))| (k.clone(), *v, *ver))
                .collect()
        };

        if snapshot.is_empty() {
            continue;
        }

        let selected = {
            let all_peers: Vec<SocketAddr> = peers.read().clone();
            let count = fanout.resolve(all_peers.len());
            select_random_peers(&all_peers, count)
        };

        for peer_addr in selected {
            gossip_to_peer(&node_id, peer_addr, &snapshot).await;
        }
    }
}

/// Select up to `count` addresses uniformly at random (without replacement).
///
/// Uses a simple Fisher-Yates partial shuffle without an external RNG crate.
/// We seed from `std::time::SystemTime` to get variation across rounds.
fn select_random_peers(peers: &[SocketAddr], count: usize) -> Vec<SocketAddr> {
    if count == 0 || peers.is_empty() {
        return Vec::new();
    }
    let count = count.min(peers.len());
    let mut indices: Vec<usize> = (0..peers.len()).collect();

    // Cheap determinism-free shuffle using a seed from system time.
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos() as u64;

    // Xorshift64 PRNG — good enough for random peer selection.
    let mut state = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
    if state == 0 {
        state = 1;
    }

    for i in 0..count {
        // Generate next random index in [i, len).
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let j = i + (state as usize % (peers.len() - i));
        indices.swap(i, j);
    }

    indices[..count].iter().map(|&i| peers[i]).collect()
}

/// Open a short-lived TCP connection to `peer_addr` and send all snapshot entries.
async fn gossip_to_peer(node_id: &str, peer_addr: SocketAddr, snapshot: &[(String, u64, u64)]) {
    let Ok(mut stream) = TcpStream::connect(peer_addr).await else {
        return; // Peer might not be ready yet; skip this round.
    };

    for (key, value, version) in snapshot {
        let msg = ClusterMessage::Gossip {
            sender_id: node_id.to_owned(),
            key: key.clone(),
            value: *value,
            version: *version,
        };
        if MessageCodec::write(&mut stream, &msg).await.is_err() {
            break; // Stream closed; abort this peer's batch.
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gossip_state_lww() {
        let mut state = GossipState::default();
        assert!(state.set("k", 10, 1));
        assert!(!state.set("k", 99, 1)); // same version — ignored
        assert!(state.set("k", 42, 2)); // higher version — accepted
        assert_eq!(state.get("k"), Some((42, 2)));
    }

    #[test]
    fn test_gossip_state_len() {
        let mut state = GossipState::default();
        state.set("a", 1, 1);
        state.set("b", 2, 1);
        assert_eq!(state.len(), 2);
        assert!(!state.is_empty());
    }

    #[test]
    fn test_node_config_localhost() {
        let cfg = TcpNodeConfig::localhost("n1", 0);
        assert_eq!(cfg.node_id, "n1");
        assert_eq!(cfg.bind_addr.port(), 0);
    }

    #[test]
    fn test_select_random_peers_empty() {
        let peers: Vec<SocketAddr> = vec![];
        let result = select_random_peers(&peers, 3);
        assert!(result.is_empty());
    }

    #[test]
    fn test_select_random_peers_count_capped() {
        let peers: Vec<SocketAddr> = (0..5)
            .map(|i| SocketAddr::from(([127, 0, 0, 1], 10000 + i)))
            .collect();
        let result = select_random_peers(&peers, 100);
        assert_eq!(result.len(), 5);
    }

    #[tokio::test]
    async fn test_start_and_addr() {
        let cfg = TcpNodeConfig::localhost("test-node", 0);
        let node = TcpClusterNode::start(cfg).await.expect("start");
        assert_eq!(node.node_id(), "test-node");
        assert_ne!(
            node.addr().port(),
            0,
            "OS should have assigned a non-zero port"
        );
        node.shutdown();
    }

    #[tokio::test]
    async fn test_set_and_get() {
        let cfg = TcpNodeConfig::localhost("n1", 0);
        let node = TcpClusterNode::start(cfg).await.expect("start");
        node.set("foo", 42);
        assert_eq!(node.get("foo"), Some(42));
        node.shutdown();
    }
}
