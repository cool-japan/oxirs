//! Gossip-based membership and failure detection.
//!
//! Each cluster node runs a [`GossipProtocol`] instance.  Nodes periodically
//! call [`GossipProtocol::tick`] to select peers to gossip with, and
//! [`GossipProtocol::receive_gossip`] to integrate information received from
//! those peers.  Suspicion and dead-state promotion happen through
//! [`GossipProtocol::update_suspicion`] and [`GossipProtocol::remove_dead`].

use std::collections::HashMap;

// ── Node status ──────────────────────────────────────────────────────────────

/// The current membership status of a cluster node.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NodeStatus {
    /// The node is considered healthy.
    Alive,
    /// The node has not been heard from for a while but has not yet been
    /// declared dead.
    Suspected,
    /// The node has been declared dead after the suspicion timeout elapsed.
    Dead,
    /// The node left the cluster voluntarily.
    Left,
}

// ── NodeInfo ────────────────────────────────────────────────────────────────

/// Metadata about a single cluster member.
#[derive(Clone, Debug)]
pub struct NodeInfo {
    /// Unique node identifier.
    pub id: String,
    /// Network address (e.g. `"127.0.0.1:7946"`).
    pub address: String,
    /// Current membership status.
    pub status: NodeStatus,
    /// Monotonically increasing heartbeat counter.
    pub heartbeat: u64,
    /// Unix timestamp (ms) when this info was last updated locally.
    pub last_seen: u64,
    /// Application-level metadata key-value pairs.
    pub metadata: HashMap<String, String>,
}

impl NodeInfo {
    fn new(id: String, address: String, now: u64) -> Self {
        Self {
            id,
            address,
            status: NodeStatus::Alive,
            heartbeat: 0,
            last_seen: now,
            metadata: HashMap::new(),
        }
    }
}

// ── Config ───────────────────────────────────────────────────────────────────

/// Tunable parameters for the gossip protocol.
#[derive(Clone, Debug)]
pub struct GossipConfig {
    /// Number of peers to gossip with per tick.
    pub fanout: usize,
    /// How often the protocol ticks (milliseconds). Informational only; the
    /// actual scheduling is the caller's responsibility.
    pub interval_ms: u64,
    /// How long (ms) without a heartbeat before a node is *suspected*.
    pub suspicion_timeout_ms: u64,
    /// How long (ms) a node must stay in `Suspected` state before it is
    /// declared *dead*.
    pub dead_timeout_ms: u64,
}

impl Default for GossipConfig {
    fn default() -> Self {
        Self {
            fanout: 3,
            interval_ms: 200,
            suspicion_timeout_ms: 1000,
            dead_timeout_ms: 5000,
        }
    }
}

// ── GossipMessage ────────────────────────────────────────────────────────────

/// A bundle of node state piggybacked on a gossip exchange.
#[derive(Clone, Debug)]
pub struct GossipMessage {
    /// The id of the node that created this message.
    pub sender_id: String,
    /// Snapshot of known nodes (or a subset).
    pub nodes: Vec<NodeInfo>,
    /// Wall-clock time (ms) when the message was created.
    pub timestamp: u64,
}

// ── GossipProtocol ───────────────────────────────────────────────────────────

/// Per-node gossip protocol state machine.
pub struct GossipProtocol {
    local_id: String,
    members: HashMap<String, NodeInfo>,
    config: GossipConfig,
    /// Monotonically increasing generation counter for this node.
    generation: u64,
}

impl GossipProtocol {
    /// Create a new protocol instance for the node identified by `local_id`.
    pub fn new(local_id: String, address: String, config: GossipConfig) -> Self {
        let mut members = HashMap::new();
        let local = NodeInfo::new(local_id.clone(), address, 0);
        members.insert(local_id.clone(), local);
        Self {
            local_id,
            members,
            config,
            generation: 0,
        }
    }

    // ── Membership ──────────────────────────────────────────────────────────

    /// Register a seed node as a known member.
    pub fn join(&mut self, seed_id: String, seed_address: String) {
        self.members
            .entry(seed_id.clone())
            .or_insert_with(|| NodeInfo::new(seed_id, seed_address, 0));
    }

    /// Mark this node as `Left` so that peers learn about the voluntary
    /// departure.
    pub fn leave(&mut self) {
        if let Some(info) = self.members.get_mut(&self.local_id) {
            info.status = NodeStatus::Left;
        }
    }

    // ── Heartbeat ────────────────────────────────────────────────────────────

    /// Increment the local node's heartbeat counter and update its
    /// `last_seen` timestamp.
    pub fn heartbeat(&mut self, current_time_ms: u64) {
        if let Some(info) = self.members.get_mut(&self.local_id) {
            info.heartbeat += 1;
            info.last_seen = current_time_ms;
        }
        self.generation += 1;
    }

    // ── Tick / gossip selection ───────────────────────────────────────────────

    /// Return the ids of `fanout` peers to gossip with this tick.
    ///
    /// Selection is deterministic (sorted) for testability.  Only `Alive`
    /// and `Suspected` peers — excluding the local node — are eligible.
    pub fn tick(&mut self, current_time_ms: u64) -> Vec<String> {
        self.heartbeat(current_time_ms);

        let mut candidates: Vec<String> = self
            .members
            .values()
            .filter(|n| {
                n.id != self.local_id
                    && (n.status == NodeStatus::Alive || n.status == NodeStatus::Suspected)
            })
            .map(|n| n.id.clone())
            .collect();

        candidates.sort();

        // Rotate the list by `generation mod len` to spread gossip load.
        if !candidates.is_empty() {
            let offset = (self.generation as usize) % candidates.len();
            candidates.rotate_left(offset);
        }

        candidates.truncate(self.config.fanout);
        candidates
    }

    // ── Receive gossip ────────────────────────────────────────────────────────

    /// Merge the [`GossipMessage`] into local state.
    ///
    /// Remote information is accepted when it carries a higher heartbeat than
    /// what is currently known, or when the remote status is more severe
    /// (`Suspected`, `Dead`, `Left`).
    pub fn receive_gossip(&mut self, msg: GossipMessage, current_time_ms: u64) {
        for remote in msg.nodes {
            if remote.id == self.local_id {
                // Never let remote nodes overwrite our own entry.
                continue;
            }

            let accept = match self.members.get(&remote.id) {
                None => true,
                Some(local) => {
                    // Accept if remote has a newer heartbeat or more severe status.
                    remote.heartbeat > local.heartbeat
                        || status_severity(&remote.status) > status_severity(&local.status)
                }
            };

            if accept {
                let mut info = remote.clone();
                info.last_seen = current_time_ms;
                self.members.insert(remote.id, info);
            }
        }
    }

    // ── Suspicion / dead promotion ────────────────────────────────────────────

    /// Promote `Alive` nodes to `Suspected` if they have not sent a
    /// heartbeat within `suspicion_timeout_ms`.
    pub fn update_suspicion(&mut self, current_time_ms: u64) {
        for info in self.members.values_mut() {
            if info.id == self.local_id {
                continue;
            }
            if info.status == NodeStatus::Alive {
                let elapsed = current_time_ms.saturating_sub(info.last_seen);
                if elapsed >= self.config.suspicion_timeout_ms {
                    info.status = NodeStatus::Suspected;
                }
            }
        }
    }

    /// Promote `Suspected` nodes to `Dead` if they have not recovered within
    /// `dead_timeout_ms` *since they were last seen*.
    pub fn promote_dead(&mut self, current_time_ms: u64) {
        for info in self.members.values_mut() {
            if info.id == self.local_id {
                continue;
            }
            if info.status == NodeStatus::Suspected {
                let elapsed = current_time_ms.saturating_sub(info.last_seen);
                if elapsed >= self.config.dead_timeout_ms {
                    info.status = NodeStatus::Dead;
                }
            }
        }
    }

    /// Remove nodes that have been `Dead` long enough (i.e. `last_seen` is
    /// older than `dead_timeout_ms` from `current_time_ms`) to keep the
    /// membership table tidy.
    pub fn remove_dead(&mut self, current_time_ms: u64) {
        self.members.retain(|id, info| {
            if *id == self.local_id {
                return true;
            }
            if info.status == NodeStatus::Dead {
                let elapsed = current_time_ms.saturating_sub(info.last_seen);
                // Remove once double the dead timeout has elapsed.
                return elapsed < self.config.dead_timeout_ms * 2;
            }
            true
        });
    }

    // ── Message creation ─────────────────────────────────────────────────────

    /// Build a [`GossipMessage`] containing a snapshot of all known members.
    pub fn create_gossip_message(&self) -> GossipMessage {
        GossipMessage {
            sender_id: self.local_id.clone(),
            nodes: self.members.values().cloned().collect(),
            timestamp: self
                .members
                .get(&self.local_id)
                .map_or(0, |n| n.last_seen),
        }
    }

    // ── Queries ──────────────────────────────────────────────────────────────

    /// All known members.
    pub fn members(&self) -> Vec<&NodeInfo> {
        self.members.values().collect()
    }

    /// Members with status `Alive`.
    pub fn alive_members(&self) -> Vec<&NodeInfo> {
        self.members
            .values()
            .filter(|n| n.status == NodeStatus::Alive)
            .collect()
    }

    /// Members with status `Suspected`.
    pub fn suspected_members(&self) -> Vec<&NodeInfo> {
        self.members
            .values()
            .filter(|n| n.status == NodeStatus::Suspected)
            .collect()
    }

    /// Members with status `Dead`.
    pub fn dead_members(&self) -> Vec<&NodeInfo> {
        self.members
            .values()
            .filter(|n| n.status == NodeStatus::Dead)
            .collect()
    }

    /// Total number of known members (including self).
    pub fn member_count(&self) -> usize {
        self.members.len()
    }

    /// Retrieve a member by id.
    pub fn get_member(&self, id: &str) -> Option<&NodeInfo> {
        self.members.get(id)
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Numeric severity of a node status (higher = worse).
fn status_severity(s: &NodeStatus) -> u8 {
    match s {
        NodeStatus::Alive => 0,
        NodeStatus::Suspected => 1,
        NodeStatus::Dead => 2,
        NodeStatus::Left => 3,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> GossipConfig {
        GossipConfig {
            fanout: 2,
            interval_ms: 100,
            suspicion_timeout_ms: 1_000,
            dead_timeout_ms: 5_000,
        }
    }

    fn node(id: &str) -> GossipProtocol {
        GossipProtocol::new(
            id.to_owned(),
            format!("127.0.0.1:700{}", id.len()),
            default_config(),
        )
    }

    // ── Construction ────────────────────────────────────────────────────────

    #[test]
    fn test_new_node_has_one_member() {
        let n = node("node1");
        assert_eq!(n.member_count(), 1);
        assert!(n.get_member("node1").is_some());
    }

    #[test]
    fn test_local_node_is_alive() {
        let n = node("node1");
        assert_eq!(
            n.get_member("node1").unwrap().status,
            NodeStatus::Alive
        );
    }

    // ── join ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_join_adds_member() {
        let mut n = node("node1");
        n.join("node2".into(), "127.0.0.1:7002".into());
        assert_eq!(n.member_count(), 2);
        assert!(n.get_member("node2").is_some());
    }

    #[test]
    fn test_join_idempotent() {
        let mut n = node("node1");
        n.join("node2".into(), "127.0.0.1:7002".into());
        n.join("node2".into(), "127.0.0.1:7002".into());
        assert_eq!(n.member_count(), 2);
    }

    #[test]
    fn test_join_preserves_address() {
        let mut n = node("node1");
        n.join("node2".into(), "10.0.0.2:9000".into());
        assert_eq!(n.get_member("node2").unwrap().address, "10.0.0.2:9000");
    }

    // ── leave ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_leave_marks_left() {
        let mut n = node("node1");
        n.leave();
        assert_eq!(
            n.get_member("node1").unwrap().status,
            NodeStatus::Left
        );
    }

    // ── heartbeat ────────────────────────────────────────────────────────────

    #[test]
    fn test_heartbeat_increments_counter() {
        let mut n = node("node1");
        n.heartbeat(1000);
        assert_eq!(n.get_member("node1").unwrap().heartbeat, 1);
    }

    #[test]
    fn test_heartbeat_updates_last_seen() {
        let mut n = node("node1");
        n.heartbeat(5000);
        assert_eq!(n.get_member("node1").unwrap().last_seen, 5000);
    }

    // ── tick ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_tick_no_peers_returns_empty() {
        let mut n = node("node1");
        let peers = n.tick(1000);
        assert!(peers.is_empty());
    }

    #[test]
    fn test_tick_respects_fanout() {
        let mut n = node("node1");
        for i in 2..=6 {
            n.join(format!("node{}", i), format!("127.0.0.1:700{}", i));
        }
        let peers = n.tick(1000);
        assert!(peers.len() <= 2);
    }

    #[test]
    fn test_tick_excludes_local_node() {
        let mut n = node("node1");
        n.join("node2".into(), "127.0.0.1:7002".into());
        let peers = n.tick(1000);
        assert!(!peers.contains(&"node1".to_owned()));
    }

    #[test]
    fn test_tick_excludes_dead_nodes() {
        let mut n = node("node1");
        n.join("dead_node".into(), "127.0.0.1:7099".into());
        if let Some(m) = n.members.get_mut("dead_node") {
            m.status = NodeStatus::Dead;
        }
        let peers = n.tick(1000);
        assert!(!peers.contains(&"dead_node".to_owned()));
    }

    #[test]
    fn test_tick_includes_suspected_nodes() {
        let mut n = node("node1");
        n.join("suspected_node".into(), "127.0.0.1:7050".into());
        if let Some(m) = n.members.get_mut("suspected_node") {
            m.status = NodeStatus::Suspected;
        }
        let peers = n.tick(1000);
        assert!(peers.contains(&"suspected_node".to_owned()));
    }

    // ── receive_gossip ────────────────────────────────────────────────────────

    #[test]
    fn test_receive_gossip_adds_new_node() {
        let mut n = node("node1");
        let msg = GossipMessage {
            sender_id: "node2".into(),
            nodes: vec![NodeInfo {
                id: "node3".into(),
                address: "127.0.0.1:7003".into(),
                status: NodeStatus::Alive,
                heartbeat: 5,
                last_seen: 0,
                metadata: HashMap::new(),
            }],
            timestamp: 0,
        };
        n.receive_gossip(msg, 1000);
        assert!(n.get_member("node3").is_some());
    }

    #[test]
    fn test_receive_gossip_updates_higher_heartbeat() {
        let mut n = node("node1");
        n.join("node2".into(), "127.0.0.1:7002".into());

        let msg = GossipMessage {
            sender_id: "node3".into(),
            nodes: vec![NodeInfo {
                id: "node2".into(),
                address: "127.0.0.1:7002".into(),
                status: NodeStatus::Alive,
                heartbeat: 99,
                last_seen: 0,
                metadata: HashMap::new(),
            }],
            timestamp: 0,
        };
        n.receive_gossip(msg, 2000);
        assert_eq!(n.get_member("node2").unwrap().heartbeat, 99);
    }

    #[test]
    fn test_receive_gossip_ignores_stale_heartbeat() {
        let mut n = node("node1");
        n.join("node2".into(), "127.0.0.1:7002".into());
        if let Some(m) = n.members.get_mut("node2") {
            m.heartbeat = 50;
        }

        let msg = GossipMessage {
            sender_id: "node3".into(),
            nodes: vec![NodeInfo {
                id: "node2".into(),
                address: "127.0.0.1:7002".into(),
                status: NodeStatus::Alive,
                heartbeat: 10, // lower than current
                last_seen: 0,
                metadata: HashMap::new(),
            }],
            timestamp: 0,
        };
        n.receive_gossip(msg, 2000);
        // Heartbeat should remain at 50.
        assert_eq!(n.get_member("node2").unwrap().heartbeat, 50);
    }

    #[test]
    fn test_receive_gossip_never_overwrites_self() {
        let mut n = node("node1");
        n.heartbeat(1000);
        let hb_before = n.get_member("node1").unwrap().heartbeat;

        let msg = GossipMessage {
            sender_id: "node2".into(),
            nodes: vec![NodeInfo {
                id: "node1".into(),
                address: "127.0.0.1:9999".into(),
                status: NodeStatus::Dead, // attacker says we're dead
                heartbeat: 999,
                last_seen: 0,
                metadata: HashMap::new(),
            }],
            timestamp: 0,
        };
        n.receive_gossip(msg, 2000);
        // Our own entry must not change.
        assert_eq!(n.get_member("node1").unwrap().heartbeat, hb_before);
        assert_eq!(
            n.get_member("node1").unwrap().status,
            NodeStatus::Alive
        );
    }

    // ── update_suspicion ─────────────────────────────────────────────────────

    #[test]
    fn test_update_suspicion_promotes_after_timeout() {
        let mut n = node("node1");
        n.join("node2".into(), "127.0.0.1:7002".into());
        // node2 last_seen = 0; suspicion timeout = 1000ms.
        n.update_suspicion(1001);
        assert_eq!(
            n.get_member("node2").unwrap().status,
            NodeStatus::Suspected
        );
    }

    #[test]
    fn test_update_suspicion_not_before_timeout() {
        let mut n = node("node1");
        n.join("node2".into(), "127.0.0.1:7002".into());
        n.update_suspicion(500);
        assert_eq!(
            n.get_member("node2").unwrap().status,
            NodeStatus::Alive
        );
    }

    #[test]
    fn test_update_suspicion_does_not_affect_self() {
        let mut n = node("node1");
        n.update_suspicion(99_999);
        assert_eq!(
            n.get_member("node1").unwrap().status,
            NodeStatus::Alive
        );
    }

    // ── promote_dead ──────────────────────────────────────────────────────────

    #[test]
    fn test_promote_dead_after_dead_timeout() {
        let mut n = node("node1");
        n.join("node2".into(), "127.0.0.1:7002".into());
        // Manually set to Suspected.
        if let Some(m) = n.members.get_mut("node2") {
            m.status = NodeStatus::Suspected;
        }
        n.promote_dead(6_000);
        assert_eq!(
            n.get_member("node2").unwrap().status,
            NodeStatus::Dead
        );
    }

    #[test]
    fn test_promote_dead_not_before_timeout() {
        let mut n = node("node1");
        n.join("node2".into(), "127.0.0.1:7002".into());
        if let Some(m) = n.members.get_mut("node2") {
            m.status = NodeStatus::Suspected;
        }
        n.promote_dead(4_999);
        assert_eq!(
            n.get_member("node2").unwrap().status,
            NodeStatus::Suspected
        );
    }

    // ── remove_dead ───────────────────────────────────────────────────────────

    #[test]
    fn test_remove_dead_cleans_old_dead_nodes() {
        let mut n = node("node1");
        n.join("node2".into(), "127.0.0.1:7002".into());
        if let Some(m) = n.members.get_mut("node2") {
            m.status = NodeStatus::Dead;
            m.last_seen = 0;
        }
        // 2× dead_timeout = 10_000ms.
        n.remove_dead(10_001);
        assert!(n.get_member("node2").is_none());
    }

    #[test]
    fn test_remove_dead_keeps_recent_dead() {
        let mut n = node("node1");
        n.join("node2".into(), "127.0.0.1:7002".into());
        if let Some(m) = n.members.get_mut("node2") {
            m.status = NodeStatus::Dead;
            m.last_seen = 8_000;
        }
        n.remove_dead(9_000); // only 1_000ms ago — keep it.
        assert!(n.get_member("node2").is_some());
    }

    // ── create_gossip_message ─────────────────────────────────────────────────

    #[test]
    fn test_create_gossip_message_includes_self() {
        let n = node("node1");
        let msg = n.create_gossip_message();
        assert!(msg.nodes.iter().any(|ni| ni.id == "node1"));
        assert_eq!(msg.sender_id, "node1");
    }

    #[test]
    fn test_create_gossip_message_includes_all_members() {
        let mut n = node("node1");
        n.join("node2".into(), "127.0.0.1:7002".into());
        n.join("node3".into(), "127.0.0.1:7003".into());
        let msg = n.create_gossip_message();
        assert_eq!(msg.nodes.len(), 3);
    }

    // ── alive / suspected / dead helpers ──────────────────────────────────────

    #[test]
    fn test_alive_members_count() {
        let mut n = node("node1");
        n.join("node2".into(), "127.0.0.1:7002".into());
        assert_eq!(n.alive_members().len(), 2);
    }

    #[test]
    fn test_suspected_members_count() {
        let mut n = node("node1");
        n.join("node2".into(), "127.0.0.1:7002".into());
        if let Some(m) = n.members.get_mut("node2") {
            m.status = NodeStatus::Suspected;
        }
        assert_eq!(n.suspected_members().len(), 1);
        assert_eq!(n.alive_members().len(), 1); // only self
    }

    #[test]
    fn test_dead_members_count() {
        let mut n = node("node1");
        n.join("node2".into(), "127.0.0.1:7002".into());
        if let Some(m) = n.members.get_mut("node2") {
            m.status = NodeStatus::Dead;
        }
        assert_eq!(n.dead_members().len(), 1);
        assert_eq!(n.alive_members().len(), 1);
    }

    // ── heartbeat propagation ────────────────────────────────────────────────

    #[test]
    fn test_heartbeat_propagated_via_gossip() {
        let mut node1 = node("node1");
        let mut node2 = GossipProtocol::new(
            "node2".into(),
            "127.0.0.1:7002".into(),
            default_config(),
        );

        // node2 increments heartbeat.
        node2.heartbeat(500);
        node2.heartbeat(1000);

        // node1 learns about node2 via gossip.
        let msg = node2.create_gossip_message();
        node1.receive_gossip(msg, 1500);

        let hb = node1.get_member("node2").unwrap().heartbeat;
        assert_eq!(hb, 2);
    }

    // ── metadata ─────────────────────────────────────────────────────────────

    #[test]
    fn test_metadata_preserved_in_gossip() {
        let mut n = node("node1");
        n.join("node2".into(), "127.0.0.1:7002".into());

        let mut node_with_meta = NodeInfo::new("node2".into(), "127.0.0.1:7002".into(), 0);
        node_with_meta.heartbeat = 1;
        node_with_meta
            .metadata
            .insert("dc".into(), "us-east".into());

        let msg = GossipMessage {
            sender_id: "node2".into(),
            nodes: vec![node_with_meta],
            timestamp: 0,
        };
        n.receive_gossip(msg, 1000);

        let dc = n
            .get_member("node2")
            .unwrap()
            .metadata
            .get("dc")
            .map(|s| s.as_str());
        assert_eq!(dc, Some("us-east"));
    }

    // ── additional coverage ──────────────────────────────────────────────────

    #[test]
    fn test_join_multiple_nodes() {
        let mut n = node("node1");
        n.join("node2".into(), "127.0.0.1:7002".into());
        n.join("node3".into(), "127.0.0.1:7003".into());
        n.join("node4".into(), "127.0.0.1:7004".into());
        assert_eq!(n.member_count(), 4);
    }

    #[test]
    fn test_heartbeat_updates_last_seen_to_timestamp() {
        let mut n = node("node1");
        n.heartbeat(9000);
        assert_eq!(n.get_member("node1").unwrap().last_seen, 9000);
    }

    #[test]
    fn test_member_count_after_join() {
        let mut n = node("node1");
        n.join("node2".into(), "127.0.0.1:7002".into());
        assert_eq!(n.member_count(), 2);
    }

    #[test]
    fn test_get_member_returns_none_for_unknown() {
        let n = node("node1");
        assert!(n.get_member("unknown").is_none());
    }

    #[test]
    fn test_node_status_equality() {
        assert_eq!(NodeStatus::Alive, NodeStatus::Alive);
        assert_ne!(NodeStatus::Alive, NodeStatus::Dead);
        assert_ne!(NodeStatus::Suspected, NodeStatus::Left);
    }

    #[test]
    fn test_gossip_config_default() {
        let cfg = GossipConfig::default();
        assert_eq!(cfg.fanout, 3);
        assert!(cfg.interval_ms > 0);
        assert!(cfg.suspicion_timeout_ms > 0);
        assert!(cfg.dead_timeout_ms > cfg.suspicion_timeout_ms);
    }

    #[test]
    fn test_receive_gossip_updates_last_seen() {
        let mut n = node("node1");
        n.join("node2".into(), "127.0.0.1:7002".into());

        let msg = GossipMessage {
            sender_id: "node3".into(),
            nodes: vec![NodeInfo {
                id: "node2".into(),
                address: "127.0.0.1:7002".into(),
                status: NodeStatus::Alive,
                heartbeat: 10,
                last_seen: 0,
                metadata: HashMap::new(),
            }],
            timestamp: 500,
        };
        n.receive_gossip(msg, 3000);
        assert_eq!(n.get_member("node2").unwrap().last_seen, 3000);
    }

    #[test]
    fn test_tick_multiple_times_rotates_peers() {
        let mut n = node("node1");
        n.join("node2".into(), "127.0.0.1:7002".into());
        n.join("node3".into(), "127.0.0.1:7003".into());
        n.join("node4".into(), "127.0.0.1:7004".into());
        // Call tick multiple times and collect the results.
        let tick1 = n.tick(1000);
        let tick2 = n.tick(2000);
        // Both should respect fanout.
        assert!(tick1.len() <= 2);
        assert!(tick2.len() <= 2);
    }

    #[test]
    fn test_alive_members_excludes_dead() {
        let mut n = node("node1");
        n.join("node2".into(), "127.0.0.1:7002".into());
        n.join("node3".into(), "127.0.0.1:7003".into());
        if let Some(m) = n.members.get_mut("node2") {
            m.status = NodeStatus::Dead;
        }
        assert_eq!(n.alive_members().len(), 2); // node1 + node3
    }

    #[test]
    fn test_left_members_not_in_alive() {
        let mut n = node("node1");
        n.join("node2".into(), "127.0.0.1:7002".into());
        if let Some(m) = n.members.get_mut("node2") {
            m.status = NodeStatus::Left;
        }
        let alive: Vec<&NodeInfo> = n
            .alive_members()
            .into_iter()
            .filter(|ni| ni.id == "node2")
            .collect();
        assert!(alive.is_empty());
    }

    #[test]
    fn test_update_suspicion_only_alive_nodes() {
        let mut n = node("node1");
        n.join("node2".into(), "127.0.0.1:7002".into());
        if let Some(m) = n.members.get_mut("node2") {
            m.status = NodeStatus::Suspected;
        }
        n.update_suspicion(99_999);
        // Already Suspected — should remain Suspected (not jump to Dead).
        assert_eq!(
            n.get_member("node2").unwrap().status,
            NodeStatus::Suspected
        );
    }

    #[test]
    fn test_gossip_message_timestamp() {
        let mut n = node("node1");
        n.heartbeat(7777);
        let msg = n.create_gossip_message();
        assert_eq!(msg.timestamp, 7777);
    }

    #[test]
    fn test_generation_increments_on_tick() {
        let mut n = node("node1");
        let gen_before = n.generation;
        n.tick(1000);
        assert!(n.generation > gen_before);
    }
}
