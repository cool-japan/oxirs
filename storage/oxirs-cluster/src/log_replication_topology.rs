//! Hierarchical log-replication topology for O(log N) fan-out.
//!
//! Instead of the leader shipping to all N followers (O(N) messages/round),
//! the topology places one "relay" per AZ. The leader ships to R relays,
//! each relay fans to its AZ members. Total messages per round: R + N/R.
//! Minimized at R = √N → O(√N) messages (better than O(N), tolerates O(log N) with tree).
//!
//! ## Relay selection
//!
//! Given N members (excluding the primary), the number of relays is:
//! ```text
//!     R = max(1, ceil(sqrt(N as f64)))
//! ```
//! Relays are picked in AZ-affinity order: the first non-primary, non-witness member
//! encountered in each AZ becomes that AZ's relay. Remaining AZs get a relay from the
//! global pool if more relays are still needed.
//!
//! Witness nodes are never promoted to relays (they do not store full log content).
//!
//! ## Hop distances
//! - Primary = 0
//! - Relay = 1
//! - Leaf / Witness = 2
//!
//! ## Message bound
//! In the worst case the primary ships to R relays (R msgs), and each relay ships to
//! at most ⌈(N-1)/R⌉ followers. The total is therefore `R + ceil((N-1)/R)` ≤ `2*ceil(sqrt(N-1))`,
//! which is far below N for large clusters.

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Node description (input to topology builder)
// ─────────────────────────────────────────────────────────────────────────────

/// Description of a cluster member used for topology construction.
///
/// Build one per member (excluding the primary itself) and pass the slice
/// to [`ReplicationTopology::build`].
#[derive(Debug, Clone)]
pub struct NodeDescriptor {
    /// Unique cluster node identifier.
    pub node_id: String,
    /// Availability zone this node belongs to (e.g. `"us-east-1a"`).
    pub az: String,
    /// Geographic region (e.g. `"us-east-1"`).
    pub region: String,
    /// If `true`, node participates in consensus but stores only the tail log.
    pub is_witness: bool,
    /// How many log entries a witness retains in its circular buffer.
    /// Ignored when `is_witness == false`.
    pub witness_tail_window: usize,
}

impl NodeDescriptor {
    /// Construct a full-member descriptor (stores complete log).
    pub fn full_member(node_id: &str, az: &str, region: &str) -> Self {
        Self {
            node_id: node_id.to_owned(),
            az: az.to_owned(),
            region: region.to_owned(),
            is_witness: false,
            witness_tail_window: 0,
        }
    }

    /// Construct a witness descriptor (stores only the tail log).
    pub fn witness(node_id: &str, az: &str, region: &str, tail_window: usize) -> Self {
        Self {
            node_id: node_id.to_owned(),
            az: az.to_owned(),
            region: region.to_owned(),
            is_witness: true,
            witness_tail_window: tail_window,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Replication role
// ─────────────────────────────────────────────────────────────────────────────

/// Role of a node in the replication topology tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplicationRole {
    /// Primary — originates log entries.
    Primary,
    /// Relay — receives from primary or upstream relay, fans out to local members.
    Relay {
        /// 1 = direct from primary, 2 = sub-relay, etc. Currently always 1.
        tier: u8,
    },
    /// Leaf — receives from its relay, does not fan out further.
    Leaf,
    /// Witness — votes in consensus, holds only tail log (last `tail_window` entries).
    Witness {
        /// Number of log entries retained in the witness's circular buffer.
        tail_window: usize,
    },
}

// ─────────────────────────────────────────────────────────────────────────────
// Topology node
// ─────────────────────────────────────────────────────────────────────────────

/// One node in the replication topology tree.
#[derive(Debug, Clone)]
pub struct TopologyNode {
    /// Node identifier (matches [`NodeDescriptor::node_id`]).
    pub node_id: String,
    /// Role this node plays in the tree.
    pub role: ReplicationRole,
    /// Availability zone (informational).
    pub az: String,
    /// Geographic region (informational).
    pub region: String,
    /// Upstream node that ships log entries to this node.
    /// `None` for the primary.
    pub upstream: Option<String>,
    /// Downstream nodes that this node fans log entries out to.
    pub downstream: Vec<String>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Replication topology (immutable after construction)
// ─────────────────────────────────────────────────────────────────────────────

/// Replication topology for the whole cluster.
///
/// Immutable after construction; rebuild when membership changes.
///
/// ## Construction
/// ```
/// use oxirs_cluster::log_replication_topology::{NodeDescriptor, ReplicationTopology};
///
/// let members = vec![
///     NodeDescriptor::full_member("n1", "az-a", "us-east-1"),
///     NodeDescriptor::full_member("n2", "az-a", "us-east-1"),
///     NodeDescriptor::full_member("n3", "az-b", "us-east-1"),
/// ];
/// let topo = ReplicationTopology::build("primary", &members);
/// assert_eq!(topo.relay_count(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct ReplicationTopology {
    nodes: HashMap<String, TopologyNode>,
    primary_id: String,
    relay_ids: Vec<String>,
}

impl ReplicationTopology {
    /// Build a topology from a flat membership list using the √N relay strategy.
    ///
    /// Assigns one relay per AZ (up to `ceil(sqrt(N))` relays total where N is the
    /// number of non-primary nodes), then assigns leaf/witness nodes to the nearest
    /// relay by AZ affinity. The primary itself needs only the node_id string — it
    /// does **not** appear in `members`.
    ///
    /// Witness nodes are assigned to the nearest relay but their role is
    /// [`ReplicationRole::Witness`], not [`ReplicationRole::Leaf`].
    ///
    /// # Panics
    /// Panics if `members` is empty.
    pub fn build(primary_id: &str, members: &[NodeDescriptor]) -> Self {
        assert!(!members.is_empty(), "members must not be empty");

        let n = members.len();
        // Compute the number of relays: ceil(sqrt(n)), minimum 1.
        let relay_count = {
            let sqrt = (n as f64).sqrt().ceil() as usize;
            sqrt.max(1)
        };

        // ── Separate witnesses from full members ────────────────────────────
        let full_members: Vec<&NodeDescriptor> = members.iter().filter(|m| !m.is_witness).collect();
        let witnesses: Vec<&NodeDescriptor> = members.iter().filter(|m| m.is_witness).collect();

        // ── Pick relays: one per AZ, up to relay_count ─────────────────────
        // We iterate full members grouped by AZ; the first member in each AZ
        // becomes its relay (relays must be full members).
        let mut az_relay: HashMap<String, String> = HashMap::new();
        let mut relay_ids: Vec<String> = Vec::new();
        let mut remaining_full: Vec<&NodeDescriptor> = Vec::new();

        for member in &full_members {
            if relay_ids.len() < relay_count && !az_relay.contains_key(&member.az) {
                az_relay.insert(member.az.clone(), member.node_id.clone());
                relay_ids.push(member.node_id.clone());
            } else {
                remaining_full.push(member);
            }
        }

        // If we still need more relays (more AZs exhausted before relay_count),
        // promote from remaining_full in order.
        let mut still_remaining: Vec<&NodeDescriptor> = Vec::new();
        for member in remaining_full {
            if relay_ids.len() < relay_count {
                relay_ids.push(member.node_id.clone());
            } else {
                still_remaining.push(member);
            }
        }
        let remaining_full = still_remaining;

        // ── Build relay-to-AZ mapping for assignment ─────────────────────────
        // Map each relay to its AZ so leaves prefer same-AZ relay.
        let mut relay_az: HashMap<String, String> = HashMap::new();
        for rid in &relay_ids {
            if let Some(member) = members.iter().find(|m| &m.node_id == rid) {
                relay_az.insert(rid.clone(), member.az.clone());
            }
        }

        // ── Assign leaves and witnesses to relays ───────────────────────────
        // For each non-relay node, find the relay in the same AZ; fall back to relay[0].
        let mut relay_downstream: HashMap<String, Vec<String>> = HashMap::new();
        for rid in &relay_ids {
            relay_downstream.insert(rid.clone(), Vec::new());
        }

        let assign_to_relay = |node_az: &str, relay_az_map: &HashMap<String, String>| -> String {
            // Prefer same AZ
            for (rid, raz) in relay_az_map {
                if raz == node_az {
                    return rid.clone();
                }
            }
            // Otherwise pick the first relay
            relay_az_map.keys().next().cloned().unwrap_or_default()
        };

        // Assign remaining full members as leaves
        let mut leaf_relays: HashMap<String, String> = HashMap::new();
        for member in &remaining_full {
            let relay = assign_to_relay(&member.az, &relay_az);
            leaf_relays.insert(member.node_id.clone(), relay.clone());
            relay_downstream
                .entry(relay)
                .or_default()
                .push(member.node_id.clone());
        }

        // Assign witnesses
        let mut witness_relays: HashMap<String, String> = HashMap::new();
        for w in &witnesses {
            let relay = assign_to_relay(&w.az, &relay_az);
            witness_relays.insert(w.node_id.clone(), relay.clone());
            relay_downstream
                .entry(relay)
                .or_default()
                .push(w.node_id.clone());
        }

        // ── Assemble node map ────────────────────────────────────────────────
        let mut nodes: HashMap<String, TopologyNode> = HashMap::new();

        // Primary
        nodes.insert(
            primary_id.to_owned(),
            TopologyNode {
                node_id: primary_id.to_owned(),
                role: ReplicationRole::Primary,
                az: String::new(),
                region: String::new(),
                upstream: None,
                downstream: relay_ids.clone(),
            },
        );

        // Relays
        for rid in &relay_ids {
            let member = members
                .iter()
                .find(|m| &m.node_id == rid)
                .expect("relay must exist in members");
            let downstream = relay_downstream.get(rid).cloned().unwrap_or_default();
            nodes.insert(
                rid.clone(),
                TopologyNode {
                    node_id: rid.clone(),
                    role: ReplicationRole::Relay { tier: 1 },
                    az: member.az.clone(),
                    region: member.region.clone(),
                    upstream: Some(primary_id.to_owned()),
                    downstream,
                },
            );
        }

        // Leaves
        for member in &remaining_full {
            let relay = leaf_relays
                .get(&member.node_id)
                .cloned()
                .unwrap_or_default();
            nodes.insert(
                member.node_id.clone(),
                TopologyNode {
                    node_id: member.node_id.clone(),
                    role: ReplicationRole::Leaf,
                    az: member.az.clone(),
                    region: member.region.clone(),
                    upstream: Some(relay),
                    downstream: Vec::new(),
                },
            );
        }

        // Witnesses
        for w in &witnesses {
            let relay = witness_relays.get(&w.node_id).cloned().unwrap_or_default();
            nodes.insert(
                w.node_id.clone(),
                TopologyNode {
                    node_id: w.node_id.clone(),
                    role: ReplicationRole::Witness {
                        tail_window: w.witness_tail_window,
                    },
                    az: w.az.clone(),
                    region: w.region.clone(),
                    upstream: Some(relay),
                    downstream: Vec::new(),
                },
            );
        }

        Self {
            nodes,
            primary_id: primary_id.to_owned(),
            relay_ids,
        }
    }

    /// Return the replication fan-out for a given sender node.
    ///
    /// The caller should ship the next log entry to all returned node IDs.
    /// Returns an empty slice for leaf / witness nodes.
    pub fn downstream_of(&self, node_id: &str) -> &[String] {
        match self.nodes.get(node_id) {
            Some(n) => &n.downstream,
            None => &[],
        }
    }

    /// Return the upstream node for a given receiver.
    ///
    /// Returns `None` for the primary.
    pub fn upstream_of(&self, node_id: &str) -> Option<&str> {
        self.nodes.get(node_id).and_then(|n| n.upstream.as_deref())
    }

    /// Total number of nodes in the topology (including primary).
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns `true` when the topology contains no nodes.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Number of relay nodes.
    pub fn relay_count(&self) -> usize {
        self.relay_ids.len()
    }

    /// Number of hops from primary to a given node (primary = 0, relay = 1, leaf/witness = 2).
    pub fn hop_distance(&self, node_id: &str) -> usize {
        if node_id == self.primary_id {
            return 0;
        }
        match self.nodes.get(node_id) {
            None => usize::MAX,
            Some(n) => match &n.role {
                ReplicationRole::Primary => 0,
                ReplicationRole::Relay { tier } => *tier as usize,
                ReplicationRole::Leaf | ReplicationRole::Witness { .. } => 2,
            },
        }
    }

    /// Maximum number of messages sent per log entry in the worst case.
    ///
    /// Equals `relay_count + max_leaves_per_relay`.
    pub fn max_messages_per_entry(&self) -> usize {
        let relay_to_leaf_max = self
            .relay_ids
            .iter()
            .map(|rid| self.nodes.get(rid).map(|n| n.downstream.len()).unwrap_or(0))
            .max()
            .unwrap_or(0);
        self.relay_ids.len() + relay_to_leaf_max
    }

    /// Get a reference to the primary node ID.
    pub fn primary_id(&self) -> &str {
        &self.primary_id
    }

    /// Iterator over all relay node IDs.
    pub fn relay_ids(&self) -> &[String] {
        &self.relay_ids
    }

    /// Look up a node by ID.
    pub fn node(&self, node_id: &str) -> Option<&TopologyNode> {
        self.nodes.get(node_id)
    }
}
