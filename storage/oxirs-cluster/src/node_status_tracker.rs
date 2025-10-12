//! # Node Status Tracking with State Machine
//!
//! Comprehensive node lifecycle management using a finite state machine.
//! Tracks node states, transitions, and provides integration with health
//! monitoring for proactive cluster management.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;
use tracing::{info, warn};

use crate::raft::OxirsNodeId;

/// Node state in the cluster lifecycle
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum NodeState {
    /// Node is initializing
    Initializing,
    /// Node is joining the cluster
    Joining,
    /// Node is active and healthy
    Active,
    /// Node is active but degraded
    Degraded,
    /// Node is suspected to have issues
    Suspect,
    /// Node has failed
    Failed,
    /// Node is gracefully leaving
    Leaving,
    /// Node has left the cluster
    Left,
    /// Node is under maintenance
    Maintenance,
}

impl NodeState {
    /// Check if the state represents an operational node
    pub fn is_operational(&self) -> bool {
        matches!(self, NodeState::Active | NodeState::Degraded)
    }

    /// Check if the state represents a problematic node
    pub fn is_problematic(&self) -> bool {
        matches!(self, NodeState::Suspect | NodeState::Failed)
    }

    /// Check if the state is transitional
    pub fn is_transitional(&self) -> bool {
        matches!(
            self,
            NodeState::Initializing | NodeState::Joining | NodeState::Leaving
        )
    }
}

/// State transition event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateTransition {
    /// Node ID
    pub node_id: OxirsNodeId,
    /// Previous state
    pub from_state: NodeState,
    /// New state
    pub to_state: NodeState,
    /// Reason for transition
    pub reason: String,
    /// Timestamp
    pub timestamp: SystemTime,
    /// Transition duration (if applicable)
    pub duration: Option<Duration>,
}

/// Node status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeStatus {
    /// Node ID
    pub node_id: OxirsNodeId,
    /// Current state
    pub current_state: NodeState,
    /// Previous state
    pub previous_state: Option<NodeState>,
    /// Time in current state
    pub time_in_state: Duration,
    /// Last state change
    pub last_state_change: SystemTime,
    /// Total state transitions
    pub transition_count: u64,
    /// Node uptime (time since Initializing)
    pub uptime: Duration,
    /// Node start time
    pub start_time: SystemTime,
    /// Additional metadata
    pub metadata: BTreeMap<String, String>,
}

/// State machine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateMachineConfig {
    /// Maximum history size
    pub max_history_size: usize,
    /// Enable automatic state transitions
    pub enable_auto_transitions: bool,
    /// Suspect timeout (seconds)
    pub suspect_timeout_secs: u64,
    /// Failed timeout (seconds)
    pub failed_timeout_secs: u64,
    /// Enable state validation
    pub enable_validation: bool,
}

impl Default for StateMachineConfig {
    fn default() -> Self {
        Self {
            max_history_size: 1000,
            enable_auto_transitions: true,
            suspect_timeout_secs: 30,
            failed_timeout_secs: 60,
            enable_validation: true,
        }
    }
}

/// State machine statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StateMachineStats {
    /// Total nodes tracked
    pub total_nodes: usize,
    /// Nodes by state
    pub nodes_by_state: BTreeMap<String, usize>,
    /// Total transitions
    pub total_transitions: u64,
    /// Invalid transition attempts
    pub invalid_transitions: u64,
    /// Average time in each state (seconds)
    pub avg_time_in_state: BTreeMap<String, f64>,
}

/// Node status tracker with state machine
pub struct NodeStatusTracker {
    config: StateMachineConfig,
    /// Node statuses
    node_statuses: Arc<RwLock<BTreeMap<OxirsNodeId, NodeStatus>>>,
    /// Transition history
    transition_history: Arc<RwLock<VecDeque<StateTransition>>>,
    /// Statistics
    stats: Arc<RwLock<StateMachineStats>>,
    /// Valid state transitions
    valid_transitions: BTreeMap<NodeState, Vec<NodeState>>,
}

impl NodeStatusTracker {
    /// Create a new node status tracker
    pub fn new(config: StateMachineConfig) -> Self {
        // Define valid state transitions
        let mut valid_transitions = BTreeMap::new();

        valid_transitions.insert(
            NodeState::Initializing,
            vec![NodeState::Joining, NodeState::Failed],
        );

        valid_transitions.insert(
            NodeState::Joining,
            vec![NodeState::Active, NodeState::Failed],
        );

        valid_transitions.insert(
            NodeState::Active,
            vec![
                NodeState::Degraded,
                NodeState::Suspect,
                NodeState::Leaving,
                NodeState::Maintenance,
            ],
        );

        valid_transitions.insert(
            NodeState::Degraded,
            vec![
                NodeState::Active,
                NodeState::Suspect,
                NodeState::Failed,
                NodeState::Leaving,
            ],
        );

        valid_transitions.insert(
            NodeState::Suspect,
            vec![NodeState::Active, NodeState::Degraded, NodeState::Failed],
        );

        valid_transitions.insert(
            NodeState::Failed,
            vec![NodeState::Initializing, NodeState::Left],
        );

        valid_transitions.insert(NodeState::Leaving, vec![NodeState::Left]);

        valid_transitions.insert(NodeState::Left, vec![NodeState::Initializing]);

        valid_transitions.insert(
            NodeState::Maintenance,
            vec![NodeState::Active, NodeState::Degraded, NodeState::Failed],
        );

        Self {
            config,
            node_statuses: Arc::new(RwLock::new(BTreeMap::new())),
            transition_history: Arc::new(RwLock::new(VecDeque::new())),
            stats: Arc::new(RwLock::new(StateMachineStats::default())),
            valid_transitions,
        }
    }

    /// Register a new node
    pub async fn register_node(&self, node_id: OxirsNodeId) {
        let now = SystemTime::now();

        let status = NodeStatus {
            node_id,
            current_state: NodeState::Initializing,
            previous_state: None,
            time_in_state: Duration::from_secs(0),
            last_state_change: now,
            transition_count: 0,
            uptime: Duration::from_secs(0),
            start_time: now,
            metadata: BTreeMap::new(),
        };

        {
            let mut statuses = self.node_statuses.write().await;
            statuses.insert(node_id, status);
        } // Drop write lock before calling update_stats

        info!("Registered node {} in state Initializing", node_id);

        self.update_stats().await;
    }

    /// Unregister a node
    pub async fn unregister_node(&self, node_id: &OxirsNodeId) {
        {
            let mut statuses = self.node_statuses.write().await;
            statuses.remove(node_id);
        } // Drop write lock before calling update_stats

        info!("Unregistered node {}", node_id);

        self.update_stats().await;
    }

    /// Transition node to a new state
    pub async fn transition_state(
        &self,
        node_id: OxirsNodeId,
        new_state: NodeState,
        reason: String,
    ) -> Result<(), String> {
        let old_state = {
            let statuses = self.node_statuses.read().await;
            statuses
                .get(&node_id)
                .map(|s| s.current_state)
                .ok_or_else(|| format!("Node {} not found", node_id))?
        };

        // Validate transition
        if self.config.enable_validation && !self.is_valid_transition(old_state, new_state) {
            let mut stats = self.stats.write().await;
            stats.invalid_transitions += 1;

            return Err(format!(
                "Invalid transition from {:?} to {:?}",
                old_state, new_state
            ));
        }

        let mut statuses = self.node_statuses.write().await;

        let status = statuses
            .get_mut(&node_id)
            .ok_or_else(|| format!("Node {} not found", node_id))?;

        // Calculate time in previous state
        let now = SystemTime::now();
        let duration = now
            .duration_since(status.last_state_change)
            .unwrap_or(Duration::from_secs(0));

        // Update status
        status.previous_state = Some(old_state);
        status.current_state = new_state;
        status.time_in_state = Duration::from_secs(0);
        status.last_state_change = now;
        status.transition_count += 1;

        // Update uptime
        status.uptime = now
            .duration_since(status.start_time)
            .unwrap_or(Duration::from_secs(0));

        // Record transition
        let transition = StateTransition {
            node_id,
            from_state: old_state,
            to_state: new_state,
            reason: reason.clone(),
            timestamp: now,
            duration: Some(duration),
        };

        drop(statuses);

        // Store transition in history
        {
            let mut history = self.transition_history.write().await;
            history.push_back(transition.clone());

            if history.len() > self.config.max_history_size {
                history.pop_front();
            }
        } // Drop write lock before calling update_stats

        info!(
            "Node {} transitioned from {:?} to {:?}: {}",
            node_id, old_state, new_state, reason
        );

        self.update_stats().await;

        Ok(())
    }

    /// Check if a state transition is valid
    fn is_valid_transition(&self, from: NodeState, to: NodeState) -> bool {
        if from == to {
            return true;
        }

        self.valid_transitions
            .get(&from)
            .map(|valid| valid.contains(&to))
            .unwrap_or(false)
    }

    /// Get node status
    pub async fn get_node_status(&self, node_id: &OxirsNodeId) -> Option<NodeStatus> {
        let statuses = self.node_statuses.read().await;
        statuses.get(node_id).cloned()
    }

    /// Get all node statuses
    pub async fn get_all_statuses(&self) -> BTreeMap<OxirsNodeId, NodeStatus> {
        self.node_statuses.read().await.clone()
    }

    /// Get nodes in a specific state
    pub async fn get_nodes_in_state(&self, state: NodeState) -> Vec<OxirsNodeId> {
        let statuses = self.node_statuses.read().await;
        statuses
            .iter()
            .filter(|(_, status)| status.current_state == state)
            .map(|(id, _)| *id)
            .collect()
    }

    /// Get transition history
    pub async fn get_transition_history(&self) -> Vec<StateTransition> {
        self.transition_history
            .read()
            .await
            .iter()
            .cloned()
            .collect()
    }

    /// Get transition history for a specific node
    pub async fn get_node_transition_history(&self, node_id: &OxirsNodeId) -> Vec<StateTransition> {
        let history = self.transition_history.read().await;
        history
            .iter()
            .filter(|t| &t.node_id == node_id)
            .cloned()
            .collect()
    }

    /// Update node metadata
    pub async fn update_metadata(
        &self,
        node_id: &OxirsNodeId,
        key: String,
        value: String,
    ) -> Result<(), String> {
        let mut statuses = self.node_statuses.write().await;

        let status = statuses
            .get_mut(node_id)
            .ok_or_else(|| format!("Node {} not found", node_id))?;

        status.metadata.insert(key, value);

        Ok(())
    }

    /// Get statistics
    pub async fn get_stats(&self) -> StateMachineStats {
        self.stats.read().await.clone()
    }

    /// Update statistics
    async fn update_stats(&self) {
        let mut stats = StateMachineStats {
            total_nodes: 0,
            nodes_by_state: BTreeMap::new(),
            total_transitions: 0,
            invalid_transitions: 0,
            avg_time_in_state: BTreeMap::new(),
        };

        // Collect data from statuses (scope read lock)
        {
            let statuses = self.node_statuses.read().await;
            stats.total_nodes = statuses.len();

            // Count nodes by state
            for status in statuses.values() {
                let state_name = format!("{:?}", status.current_state);
                *stats.nodes_by_state.entry(state_name).or_insert(0) += 1;
                stats.total_transitions += status.transition_count;
            }
        } // Drop statuses read lock

        // Calculate average time in each state (scope read lock)
        {
            let history = self.transition_history.read().await;
            let mut state_durations: BTreeMap<String, Vec<u64>> = BTreeMap::new();

            for transition in history.iter() {
                if let Some(duration) = transition.duration {
                    let state_name = format!("{:?}", transition.from_state);
                    state_durations
                        .entry(state_name)
                        .or_insert_with(Vec::new)
                        .push(duration.as_secs());
                }
            }

            for (state, durations) in state_durations.iter() {
                if !durations.is_empty() {
                    let avg = durations.iter().sum::<u64>() as f64 / durations.len() as f64;
                    stats.avg_time_in_state.insert(state.clone(), avg);
                }
            }
        } // Drop history read lock

        // Preserve invalid transitions count and update (scope locks separately)
        {
            let old_stats = self.stats.read().await;
            stats.invalid_transitions = old_stats.invalid_transitions;
        } // Drop stats read lock before acquiring write lock

        *self.stats.write().await = stats;
    }

    /// Perform automatic state checks
    pub async fn perform_auto_checks(&self) {
        if !self.config.enable_auto_transitions {
            return;
        }

        let statuses = self.node_statuses.read().await;
        let now = SystemTime::now();
        let mut transitions = Vec::new();

        for (node_id, status) in statuses.iter() {
            let time_in_state = now
                .duration_since(status.last_state_change)
                .unwrap_or(Duration::from_secs(0));

            // Check for suspect timeout
            if status.current_state == NodeState::Suspect
                && time_in_state.as_secs() > self.config.suspect_timeout_secs
            {
                transitions.push((
                    *node_id,
                    NodeState::Failed,
                    "Suspect timeout exceeded".to_string(),
                ));
            }

            // Check for failed timeout
            if status.current_state == NodeState::Failed
                && time_in_state.as_secs() > self.config.failed_timeout_secs
            {
                transitions.push((
                    *node_id,
                    NodeState::Left,
                    "Failed timeout exceeded".to_string(),
                ));
            }
        }

        drop(statuses);

        // Apply transitions
        for (node_id, new_state, reason) in transitions {
            if let Err(e) = self.transition_state(node_id, new_state, reason).await {
                warn!("Auto-transition failed for node {}: {}", node_id, e);
            }
        }
    }

    /// Clear all data
    pub async fn clear(&self) {
        self.node_statuses.write().await.clear();
        self.transition_history.write().await.clear();
        *self.stats.write().await = StateMachineStats::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_node_status_tracker_creation() {
        let config = StateMachineConfig::default();
        let tracker = NodeStatusTracker::new(config);

        let stats = tracker.get_stats().await;
        assert_eq!(stats.total_nodes, 0);
    }

    #[tokio::test]
    async fn test_register_node() {
        let config = StateMachineConfig::default();
        let tracker = NodeStatusTracker::new(config);

        tracker.register_node(1).await;

        let status = tracker.get_node_status(&1).await;
        assert!(status.is_some());

        let status = status.unwrap();
        assert_eq!(status.current_state, NodeState::Initializing);
        assert_eq!(status.transition_count, 0);
    }

    #[tokio::test]
    async fn test_valid_transition() {
        let config = StateMachineConfig::default();
        let tracker = NodeStatusTracker::new(config);

        tracker.register_node(1).await;

        let result = tracker
            .transition_state(1, NodeState::Joining, "Starting join process".to_string())
            .await;

        assert!(result.is_ok());

        let status = tracker.get_node_status(&1).await.unwrap();
        assert_eq!(status.current_state, NodeState::Joining);
        assert_eq!(status.previous_state, Some(NodeState::Initializing));
        assert_eq!(status.transition_count, 1);
    }

    #[tokio::test]
    async fn test_invalid_transition() {
        let config = StateMachineConfig::default();
        let tracker = NodeStatusTracker::new(config);

        tracker.register_node(1).await;

        let result = tracker
            .transition_state(1, NodeState::Active, "Invalid transition".to_string())
            .await;

        assert!(result.is_err());

        let stats = tracker.get_stats().await;
        assert_eq!(stats.invalid_transitions, 1);
    }

    #[tokio::test]
    async fn test_state_machine_flow() {
        let config = StateMachineConfig::default();
        let tracker = NodeStatusTracker::new(config);

        tracker.register_node(1).await;

        // Initializing -> Joining
        tracker
            .transition_state(1, NodeState::Joining, "Joining cluster".to_string())
            .await
            .unwrap();

        // Joining -> Active
        tracker
            .transition_state(1, NodeState::Active, "Joined successfully".to_string())
            .await
            .unwrap();

        // Active -> Degraded
        tracker
            .transition_state(1, NodeState::Degraded, "Performance degraded".to_string())
            .await
            .unwrap();

        let status = tracker.get_node_status(&1).await.unwrap();
        assert_eq!(status.current_state, NodeState::Degraded);
        assert_eq!(status.transition_count, 3);
    }

    #[tokio::test]
    async fn test_get_nodes_in_state() {
        let config = StateMachineConfig::default();
        let tracker = NodeStatusTracker::new(config);

        tracker.register_node(1).await;
        tracker.register_node(2).await;
        tracker.register_node(3).await;

        tracker
            .transition_state(1, NodeState::Joining, "test".to_string())
            .await
            .unwrap();
        tracker
            .transition_state(2, NodeState::Joining, "test".to_string())
            .await
            .unwrap();

        let joining_nodes = tracker.get_nodes_in_state(NodeState::Joining).await;
        assert_eq!(joining_nodes.len(), 2);
        assert!(joining_nodes.contains(&1));
        assert!(joining_nodes.contains(&2));
    }

    #[tokio::test]
    async fn test_transition_history() {
        let config = StateMachineConfig::default();
        let tracker = NodeStatusTracker::new(config);

        tracker.register_node(1).await;

        tracker
            .transition_state(1, NodeState::Joining, "test1".to_string())
            .await
            .unwrap();
        tracker
            .transition_state(1, NodeState::Active, "test2".to_string())
            .await
            .unwrap();

        let history = tracker.get_node_transition_history(&1).await;
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].from_state, NodeState::Initializing);
        assert_eq!(history[0].to_state, NodeState::Joining);
        assert_eq!(history[1].from_state, NodeState::Joining);
        assert_eq!(history[1].to_state, NodeState::Active);
    }

    #[tokio::test]
    async fn test_metadata() {
        let config = StateMachineConfig::default();
        let tracker = NodeStatusTracker::new(config);

        tracker.register_node(1).await;

        tracker
            .update_metadata(&1, "version".to_string(), "1.0.0".to_string())
            .await
            .unwrap();
        tracker
            .update_metadata(&1, "region".to_string(), "us-west".to_string())
            .await
            .unwrap();

        let status = tracker.get_node_status(&1).await.unwrap();
        assert_eq!(status.metadata.get("version"), Some(&"1.0.0".to_string()));
        assert_eq!(status.metadata.get("region"), Some(&"us-west".to_string()));
    }

    #[tokio::test]
    async fn test_stats() {
        let config = StateMachineConfig::default();
        let tracker = NodeStatusTracker::new(config);

        tracker.register_node(1).await;
        tracker.register_node(2).await;
        tracker.register_node(3).await;

        tracker
            .transition_state(1, NodeState::Joining, "test".to_string())
            .await
            .unwrap();
        tracker
            .transition_state(2, NodeState::Joining, "test".to_string())
            .await
            .unwrap();
        tracker
            .transition_state(1, NodeState::Active, "test".to_string())
            .await
            .unwrap();

        let stats = tracker.get_stats().await;
        assert_eq!(stats.total_nodes, 3);
        assert_eq!(stats.total_transitions, 3);
    }

    #[tokio::test]
    async fn test_unregister_node() {
        let config = StateMachineConfig::default();
        let tracker = NodeStatusTracker::new(config);

        tracker.register_node(1).await;
        assert!(tracker.get_node_status(&1).await.is_some());

        tracker.unregister_node(&1).await;
        assert!(tracker.get_node_status(&1).await.is_none());
    }

    #[tokio::test]
    async fn test_node_state_helpers() {
        assert!(NodeState::Active.is_operational());
        assert!(NodeState::Degraded.is_operational());
        assert!(!NodeState::Failed.is_operational());

        assert!(NodeState::Suspect.is_problematic());
        assert!(NodeState::Failed.is_problematic());
        assert!(!NodeState::Active.is_problematic());

        assert!(NodeState::Initializing.is_transitional());
        assert!(NodeState::Joining.is_transitional());
        assert!(!NodeState::Active.is_transitional());
    }

    #[tokio::test]
    async fn test_clear() {
        let config = StateMachineConfig::default();
        let tracker = NodeStatusTracker::new(config);

        tracker.register_node(1).await;
        tracker.register_node(2).await;

        tracker.clear().await;

        let stats = tracker.get_stats().await;
        assert_eq!(stats.total_nodes, 0);
    }
}
