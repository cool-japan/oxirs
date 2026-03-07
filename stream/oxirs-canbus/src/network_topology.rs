//! CAN bus network topology modeling.
//!
//! Provides node-and-edge graph representation of a CAN bus network,
//! with sender/receiver lookups and BFS shortest-path routing.

use std::collections::{HashMap, HashSet, VecDeque};

// ---------------------------------------------------------------------------
// Node type
// ---------------------------------------------------------------------------

/// The role of a node in the CAN bus network.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NodeType {
    /// Electronic Control Unit (general ECU).
    Ecu,
    /// Gateway between two bus segments.
    Gateway,
    /// Sensing device.
    Sensor,
    /// Actuating device.
    Actuator,
    /// Diagnostic tool or interface.
    Diagnostic,
}

// ---------------------------------------------------------------------------
// CanNode
// ---------------------------------------------------------------------------

/// A node in the CAN bus network.
#[derive(Debug, Clone)]
pub struct CanNode {
    /// Unique identifier for this node (e.g. `"ECU_01"`).
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Role of this node.
    pub node_type: NodeType,
    /// CAN arbitration ID range this node "owns" `(min, max)` (inclusive).
    pub can_id_range: (u32, u32),
    /// CAN IDs that this node transmits.
    pub sends: Vec<u32>,
    /// CAN IDs that this node receives.
    pub receives: Vec<u32>,
}

// ---------------------------------------------------------------------------
// Topology error
// ---------------------------------------------------------------------------

/// Errors returned by [`NetworkTopology`] operations.
#[derive(Debug, PartialEq, Eq)]
pub enum TopologyError {
    /// A referenced node was not found in the topology.
    NodeNotFound(String),
    /// The requested connection already exists.
    AlreadyConnected,
}

impl std::fmt::Display for TopologyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NodeNotFound(id) => write!(f, "node not found: {id}"),
            Self::AlreadyConnected => write!(f, "nodes are already connected"),
        }
    }
}

impl std::error::Error for TopologyError {}

// ---------------------------------------------------------------------------
// NetworkTopology
// ---------------------------------------------------------------------------

/// An undirected graph of CAN bus nodes and their physical connections.
pub struct NetworkTopology {
    nodes: HashMap<String, CanNode>,
    /// Set of undirected edges stored as `(min_id, max_id)` to avoid
    /// ordering duplicates.
    connections: Vec<(String, String)>,
}

impl Default for NetworkTopology {
    fn default() -> Self {
        Self::new()
    }
}

impl NetworkTopology {
    /// Create an empty topology.
    pub fn new() -> Self {
        NetworkTopology {
            nodes: HashMap::new(),
            connections: Vec::new(),
        }
    }

    /// Add a node.  If a node with the same id already exists it is replaced.
    pub fn add_node(&mut self, node: CanNode) {
        self.nodes.insert(node.id.clone(), node);
    }

    /// Remove the node with `id` and all its connections.
    /// Returns `true` if the node existed.
    pub fn remove_node(&mut self, id: &str) -> bool {
        if self.nodes.remove(id).is_some() {
            self.connections.retain(|(a, b)| a != id && b != id);
            true
        } else {
            false
        }
    }

    /// Add an undirected connection between nodes `from` and `to`.
    ///
    /// Returns [`TopologyError::NodeNotFound`] if either node is absent.
    /// Returns [`TopologyError::AlreadyConnected`] if the edge already exists.
    pub fn connect(&mut self, from: &str, to: &str) -> Result<(), TopologyError> {
        if !self.nodes.contains_key(from) {
            return Err(TopologyError::NodeNotFound(from.to_string()));
        }
        if !self.nodes.contains_key(to) {
            return Err(TopologyError::NodeNotFound(to.to_string()));
        }
        // Canonical ordering to detect duplicates regardless of direction.
        let (a, b) = canonical_edge(from, to);
        if self.connections.iter().any(|(x, y)| x == &a && y == &b) {
            return Err(TopologyError::AlreadyConnected);
        }
        self.connections.push((a, b));
        Ok(())
    }

    /// Return references to all nodes directly connected to the node with `id`.
    pub fn connected_nodes(&self, id: &str) -> Vec<&CanNode> {
        let mut neighbours = Vec::new();
        for (a, b) in &self.connections {
            if a == id {
                if let Some(node) = self.nodes.get(b) {
                    neighbours.push(node);
                }
            } else if b == id {
                if let Some(node) = self.nodes.get(a) {
                    neighbours.push(node);
                }
            }
        }
        neighbours
    }

    /// All nodes that list `can_id` in their `sends` list.
    pub fn senders_of(&self, can_id: u32) -> Vec<&CanNode> {
        self.nodes
            .values()
            .filter(|n| n.sends.contains(&can_id))
            .collect()
    }

    /// All nodes that list `can_id` in their `receives` list.
    pub fn receivers_of(&self, can_id: u32) -> Vec<&CanNode> {
        self.nodes
            .values()
            .filter(|n| n.receives.contains(&can_id))
            .collect()
    }

    /// BFS shortest path from `from` to `to`.
    ///
    /// Returns `Some(path)` where `path` is the ordered sequence of node IDs
    /// (inclusive of both endpoints), or `None` if no path exists.
    pub fn route(&self, from: &str, to: &str) -> Option<Vec<String>> {
        if !self.nodes.contains_key(from) || !self.nodes.contains_key(to) {
            return None;
        }
        if from == to {
            return Some(vec![from.to_string()]);
        }

        let mut visited: HashSet<&str> = HashSet::new();
        let mut queue: VecDeque<(String, Vec<String>)> = VecDeque::new();
        queue.push_back((from.to_string(), vec![from.to_string()]));
        visited.insert(from);

        while let Some((current, path)) = queue.pop_front() {
            for neighbour in self.connected_nodes(&current) {
                let nid = neighbour.id.as_str();
                if !visited.contains(nid) {
                    let mut new_path = path.clone();
                    new_path.push(nid.to_string());
                    if nid == to {
                        return Some(new_path);
                    }
                    visited.insert(nid);
                    queue.push_back((nid.to_string(), new_path));
                }
            }
        }
        None
    }

    /// Number of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of connections (edges).
    pub fn connection_count(&self) -> usize {
        self.connections.len()
    }

    /// All gateway nodes.
    pub fn gateways(&self) -> Vec<&CanNode> {
        self.nodes
            .values()
            .filter(|n| n.node_type == NodeType::Gateway)
            .collect()
    }

    /// `true` if every node in the topology can reach every other node
    /// (the underlying undirected graph is connected).
    ///
    /// An empty or single-node topology is considered connected.
    pub fn is_connected(&self) -> bool {
        if self.nodes.len() <= 1 {
            return true;
        }

        // Pick any starting node.
        let start = self.nodes.keys().next().expect("node count >= 1");
        let mut visited: HashSet<&str> = HashSet::new();
        let mut stack: Vec<&str> = vec![start.as_str()];
        visited.insert(start.as_str());

        while let Some(current) = stack.pop() {
            for neighbour in self.connected_nodes(current) {
                let nid = neighbour.id.as_str();
                if !visited.contains(nid) {
                    visited.insert(nid);
                    stack.push(nid);
                }
            }
        }

        visited.len() == self.nodes.len()
    }
}

/// Return a canonical (sorted) pair for an undirected edge.
fn canonical_edge(a: &str, b: &str) -> (String, String) {
    if a <= b {
        (a.to_string(), b.to_string())
    } else {
        (b.to_string(), a.to_string())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_node(id: &str, node_type: NodeType) -> CanNode {
        CanNode {
            id: id.to_string(),
            name: format!("Node {id}"),
            node_type,
            can_id_range: (0, 0x7FF),
            sends: Vec::new(),
            receives: Vec::new(),
        }
    }

    fn make_sender(id: &str, sends: &[u32]) -> CanNode {
        CanNode {
            id: id.to_string(),
            name: id.to_string(),
            node_type: NodeType::Ecu,
            can_id_range: (0, 0x7FF),
            sends: sends.to_vec(),
            receives: Vec::new(),
        }
    }

    fn make_receiver(id: &str, receives: &[u32]) -> CanNode {
        CanNode {
            id: id.to_string(),
            name: id.to_string(),
            node_type: NodeType::Ecu,
            can_id_range: (0, 0x7FF),
            sends: Vec::new(),
            receives: receives.to_vec(),
        }
    }

    // --- add_node ---

    #[test]
    fn test_add_node_increments_count() {
        let mut topo = NetworkTopology::new();
        topo.add_node(make_node("A", NodeType::Ecu));
        assert_eq!(topo.node_count(), 1);
    }

    #[test]
    fn test_add_multiple_nodes() {
        let mut topo = NetworkTopology::new();
        topo.add_node(make_node("A", NodeType::Ecu));
        topo.add_node(make_node("B", NodeType::Gateway));
        assert_eq!(topo.node_count(), 2);
    }

    #[test]
    fn test_add_node_replaces_existing() {
        let mut topo = NetworkTopology::new();
        topo.add_node(make_node("A", NodeType::Ecu));
        topo.add_node(make_node("A", NodeType::Gateway)); // Replace
        assert_eq!(topo.node_count(), 1);
    }

    // --- remove_node ---

    #[test]
    fn test_remove_node_returns_true() {
        let mut topo = NetworkTopology::new();
        topo.add_node(make_node("A", NodeType::Ecu));
        assert!(topo.remove_node("A"));
        assert_eq!(topo.node_count(), 0);
    }

    #[test]
    fn test_remove_nonexistent_node_returns_false() {
        let mut topo = NetworkTopology::new();
        assert!(!topo.remove_node("X"));
    }

    #[test]
    fn test_remove_node_cleans_connections() {
        let mut topo = NetworkTopology::new();
        topo.add_node(make_node("A", NodeType::Ecu));
        topo.add_node(make_node("B", NodeType::Ecu));
        topo.connect("A", "B").expect("connect should succeed");
        topo.remove_node("A");
        assert_eq!(topo.connection_count(), 0);
    }

    // --- connect ---

    #[test]
    fn test_connect_success() {
        let mut topo = NetworkTopology::new();
        topo.add_node(make_node("A", NodeType::Ecu));
        topo.add_node(make_node("B", NodeType::Ecu));
        assert!(topo.connect("A", "B").is_ok());
        assert_eq!(topo.connection_count(), 1);
    }

    #[test]
    fn test_connect_node_not_found() {
        let mut topo = NetworkTopology::new();
        topo.add_node(make_node("A", NodeType::Ecu));
        let err = topo.connect("A", "MISSING").unwrap_err();
        assert_eq!(err, TopologyError::NodeNotFound("MISSING".into()));
    }

    #[test]
    fn test_connect_already_connected() {
        let mut topo = NetworkTopology::new();
        topo.add_node(make_node("A", NodeType::Ecu));
        topo.add_node(make_node("B", NodeType::Ecu));
        topo.connect("A", "B")
            .expect("first connect should succeed");
        let err = topo.connect("A", "B").unwrap_err();
        assert_eq!(err, TopologyError::AlreadyConnected);
    }

    #[test]
    fn test_connect_already_connected_reversed() {
        let mut topo = NetworkTopology::new();
        topo.add_node(make_node("A", NodeType::Ecu));
        topo.add_node(make_node("B", NodeType::Ecu));
        topo.connect("A", "B")
            .expect("first connect should succeed");
        let err = topo.connect("B", "A").unwrap_err();
        assert_eq!(err, TopologyError::AlreadyConnected);
    }

    // --- connected_nodes ---

    #[test]
    fn test_connected_nodes_returns_neighbours() {
        let mut topo = NetworkTopology::new();
        topo.add_node(make_node("A", NodeType::Ecu));
        topo.add_node(make_node("B", NodeType::Ecu));
        topo.add_node(make_node("C", NodeType::Ecu));
        topo.connect("A", "B").expect("connect should succeed");
        topo.connect("A", "C").expect("connect should succeed");
        let n = topo.connected_nodes("A");
        assert_eq!(n.len(), 2);
    }

    #[test]
    fn test_connected_nodes_empty_for_isolated_node() {
        let mut topo = NetworkTopology::new();
        topo.add_node(make_node("A", NodeType::Ecu));
        assert!(topo.connected_nodes("A").is_empty());
    }

    // --- senders_of / receivers_of ---

    #[test]
    fn test_senders_of() {
        let mut topo = NetworkTopology::new();
        topo.add_node(make_sender("A", &[0x100, 0x200]));
        topo.add_node(make_sender("B", &[0x100]));
        let senders = topo.senders_of(0x100);
        assert_eq!(senders.len(), 2);
    }

    #[test]
    fn test_senders_of_no_match() {
        let mut topo = NetworkTopology::new();
        topo.add_node(make_sender("A", &[0x100]));
        assert!(topo.senders_of(0x999).is_empty());
    }

    #[test]
    fn test_receivers_of() {
        let mut topo = NetworkTopology::new();
        topo.add_node(make_receiver("A", &[0x100]));
        topo.add_node(make_receiver("B", &[0x200]));
        let receivers = topo.receivers_of(0x100);
        assert_eq!(receivers.len(), 1);
        assert_eq!(receivers[0].id, "A");
    }

    // --- route (BFS) ---

    #[test]
    fn test_route_direct() {
        let mut topo = NetworkTopology::new();
        topo.add_node(make_node("A", NodeType::Ecu));
        topo.add_node(make_node("B", NodeType::Ecu));
        topo.connect("A", "B").expect("connect should succeed");
        let path = topo.route("A", "B").expect("path should exist");
        assert_eq!(path, vec!["A", "B"]);
    }

    #[test]
    fn test_route_indirect() {
        let mut topo = NetworkTopology::new();
        topo.add_node(make_node("A", NodeType::Ecu));
        topo.add_node(make_node("B", NodeType::Gateway));
        topo.add_node(make_node("C", NodeType::Ecu));
        topo.connect("A", "B").expect("connect should succeed");
        topo.connect("B", "C").expect("connect should succeed");
        let path = topo.route("A", "C").expect("path should exist");
        assert_eq!(path, vec!["A", "B", "C"]);
    }

    #[test]
    fn test_route_same_node() {
        let mut topo = NetworkTopology::new();
        topo.add_node(make_node("A", NodeType::Ecu));
        let path = topo.route("A", "A").expect("path should exist");
        assert_eq!(path, vec!["A"]);
    }

    #[test]
    fn test_route_returns_none_if_disconnected() {
        let mut topo = NetworkTopology::new();
        topo.add_node(make_node("A", NodeType::Ecu));
        topo.add_node(make_node("B", NodeType::Ecu));
        // No connection added
        assert!(topo.route("A", "B").is_none());
    }

    #[test]
    fn test_route_returns_none_if_node_missing() {
        let mut topo = NetworkTopology::new();
        topo.add_node(make_node("A", NodeType::Ecu));
        assert!(topo.route("A", "GHOST").is_none());
    }

    // --- gateways ---

    #[test]
    fn test_gateways_filter() {
        let mut topo = NetworkTopology::new();
        topo.add_node(make_node("A", NodeType::Ecu));
        topo.add_node(make_node("B", NodeType::Gateway));
        topo.add_node(make_node("C", NodeType::Gateway));
        assert_eq!(topo.gateways().len(), 2);
    }

    #[test]
    fn test_gateways_empty_when_none() {
        let mut topo = NetworkTopology::new();
        topo.add_node(make_node("A", NodeType::Ecu));
        assert!(topo.gateways().is_empty());
    }

    // --- is_connected ---

    #[test]
    fn test_is_connected_true_when_all_connected() {
        let mut topo = NetworkTopology::new();
        topo.add_node(make_node("A", NodeType::Ecu));
        topo.add_node(make_node("B", NodeType::Ecu));
        topo.add_node(make_node("C", NodeType::Ecu));
        topo.connect("A", "B").expect("connect should succeed");
        topo.connect("B", "C").expect("connect should succeed");
        assert!(topo.is_connected());
    }

    #[test]
    fn test_is_connected_false_when_isolated_node() {
        let mut topo = NetworkTopology::new();
        topo.add_node(make_node("A", NodeType::Ecu));
        topo.add_node(make_node("B", NodeType::Ecu)); // isolated
        topo.add_node(make_node("C", NodeType::Ecu));
        topo.connect("A", "C").expect("connect should succeed");
        assert!(!topo.is_connected());
    }

    #[test]
    fn test_is_connected_empty_topology() {
        let topo = NetworkTopology::new();
        assert!(topo.is_connected());
    }

    #[test]
    fn test_is_connected_single_node() {
        let mut topo = NetworkTopology::new();
        topo.add_node(make_node("A", NodeType::Ecu));
        assert!(topo.is_connected());
    }

    // --- node_count / connection_count ---

    #[test]
    fn test_node_count() {
        let mut topo = NetworkTopology::new();
        topo.add_node(make_node("A", NodeType::Ecu));
        topo.add_node(make_node("B", NodeType::Sensor));
        assert_eq!(topo.node_count(), 2);
    }

    #[test]
    fn test_connection_count() {
        let mut topo = NetworkTopology::new();
        topo.add_node(make_node("A", NodeType::Ecu));
        topo.add_node(make_node("B", NodeType::Ecu));
        topo.add_node(make_node("C", NodeType::Ecu));
        topo.connect("A", "B").expect("connect should succeed");
        topo.connect("B", "C").expect("connect should succeed");
        assert_eq!(topo.connection_count(), 2);
    }

    // --- NodeType variants ---

    #[test]
    fn test_node_type_all_variants_reachable() {
        let types = [
            NodeType::Ecu,
            NodeType::Gateway,
            NodeType::Sensor,
            NodeType::Actuator,
            NodeType::Diagnostic,
        ];
        // Just ensure all variants can be constructed and compared.
        assert_ne!(types[0], types[1]);
        assert_ne!(types[2], types[3]);
    }

    #[test]
    fn test_topology_error_display() {
        let err = TopologyError::NodeNotFound("X".into());
        assert!(err.to_string().contains("X"));
        let err2 = TopologyError::AlreadyConnected;
        assert!(!err2.to_string().is_empty());
    }

    #[test]
    fn test_default_topology_is_empty() {
        let topo = NetworkTopology::default();
        assert_eq!(topo.node_count(), 0);
        assert_eq!(topo.connection_count(), 0);
    }
}
