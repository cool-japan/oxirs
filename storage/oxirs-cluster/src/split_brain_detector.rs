//! Split-brain detection for distributed clusters.
//!
//! Detects network partitions by analysing each node's view of the cluster,
//! identifies disconnected groups, and recommends remediation actions.

use std::collections::HashMap;

/// The observed liveness state of a node.
#[derive(Debug, Clone, PartialEq)]
pub enum NodeState {
    Active,
    Suspected,
    Unreachable,
    Partitioned,
}

/// A single node's view of the entire cluster at a point in time.
#[derive(Debug, Clone)]
pub struct ClusterView {
    pub node_id: String,
    pub known_nodes: HashMap<String, NodeState>,
    pub last_updated: u64,
}

impl ClusterView {
    pub fn new(node_id: impl Into<String>, last_updated: u64) -> Self {
        ClusterView {
            node_id: node_id.into(),
            known_nodes: HashMap::new(),
            last_updated,
        }
    }

    pub fn with_node(mut self, id: impl Into<String>, state: NodeState) -> Self {
        self.known_nodes.insert(id.into(), state);
        self
    }
}

/// Describes a detected partition event.
#[derive(Debug, Clone)]
pub struct PartitionReport {
    pub detected_at: u64,
    pub partition_a: Vec<String>,
    pub partition_b: Vec<String>,
    pub confidence: f64,
}

/// Recommended action to take in response to split-brain.
#[derive(Debug, Clone, PartialEq)]
pub enum SplitBrainAction {
    /// Cluster is healthy.
    NoAction,
    /// Partitioned but within tolerable bounds — wait for self-healing.
    WaitForHealing,
    /// Promote a specific node as primary to break the tie.
    ForcePrimary(String),
    /// Cluster is unrecoverable; shut down non-primary partitions.
    Shutdown,
}

/// Detects split-brain scenarios in distributed clusters.
pub struct SplitBrainDetector {
    quorum_size: usize,
    views: HashMap<String, ClusterView>,
    partition_timeout_ms: u64,
}

impl SplitBrainDetector {
    /// Create a new detector.
    ///
    /// * `quorum_size` — minimum node count for a healthy quorum.
    /// * `partition_timeout_ms` — age beyond which a view is considered stale.
    pub fn new(quorum_size: usize, partition_timeout_ms: u64) -> Self {
        SplitBrainDetector {
            quorum_size,
            views: HashMap::new(),
            partition_timeout_ms,
        }
    }

    /// Submit a cluster view from a node.
    pub fn update_view(&mut self, view: ClusterView) {
        self.views.insert(view.node_id.clone(), view);
    }

    /// Return the number of nodes that are currently `Active`.
    pub fn active_node_count(&self) -> usize {
        self.views
            .values()
            .filter(|v| {
                v.known_nodes
                    .get(&v.node_id)
                    .map(|s| *s == NodeState::Active)
                    .unwrap_or(true) // treat self as active if not listed
            })
            .count()
    }

    /// Detect partitions by finding groups of nodes that each consider the
    /// other group unreachable or partitioned.
    pub fn detect_partitions(&self) -> Vec<PartitionReport> {
        let mut reports: Vec<PartitionReport> = Vec::new();
        let node_ids: Vec<String> = self.views.keys().cloned().collect();
        let total = node_ids.len();
        if total < 2 {
            return reports;
        }

        // Build a "can see" adjacency: node A can see node B if A's view lists B as Active or Suspected.
        let can_see = |a: &str, b: &str| -> bool {
            if let Some(view) = self.views.get(a) {
                match view.known_nodes.get(b) {
                    Some(NodeState::Active) | Some(NodeState::Suspected) | None => true,
                    Some(NodeState::Unreachable) | Some(NodeState::Partitioned) => false,
                }
            } else {
                false
            }
        };

        // Find pairs of nodes that cannot see each other (mutual blindness)
        let mut partition_pairs: Vec<(&str, &str)> = Vec::new();
        for i in 0..node_ids.len() {
            for j in (i + 1)..node_ids.len() {
                let a = &node_ids[i];
                let b = &node_ids[j];
                if !can_see(a, b) && !can_see(b, a) {
                    partition_pairs.push((a.as_str(), b.as_str()));
                }
            }
        }

        if partition_pairs.is_empty() {
            return reports;
        }

        // Simple two-group partition detection: group by mutual visibility
        // Using union-find-style grouping
        let groups = find_connected_groups(&node_ids, &self.views, can_see);

        if groups.len() < 2 {
            return reports;
        }

        let now = self
            .views
            .values()
            .map(|v| v.last_updated)
            .max()
            .unwrap_or(0);

        // Report each adjacent pair of groups
        for i in 0..groups.len() {
            for j in (i + 1)..groups.len() {
                let partition_a = groups[i].clone();
                let partition_b = groups[j].clone();
                let confidence = compute_confidence(&partition_a, &partition_b, total);
                reports.push(PartitionReport {
                    detected_at: now,
                    partition_a,
                    partition_b,
                    confidence,
                });
            }
        }

        reports
    }

    /// Check whether the given node set forms a quorum.
    pub fn has_quorum(&self, node_ids: &[String]) -> bool {
        let active_count = node_ids
            .iter()
            .filter(|id| {
                self.views
                    .get(*id)
                    .map(|v| {
                        v.known_nodes
                            .get(*id)
                            .map(|s| matches!(s, NodeState::Active | NodeState::Suspected))
                            .unwrap_or(true)
                    })
                    .unwrap_or(false)
            })
            .count();
        active_count >= self.quorum_size
    }

    /// Recommend a remediation action.
    pub fn recommend_action(&self) -> SplitBrainAction {
        let partitions = self.detect_partitions();
        if partitions.is_empty() {
            return SplitBrainAction::NoAction;
        }

        // Find largest partition
        let largest = partitions
            .iter()
            .map(|p| p.partition_a.len().max(p.partition_b.len()))
            .max()
            .unwrap_or(0);

        if largest >= self.quorum_size {
            // Promote the lexicographically first node in the largest partition as primary
            let primary_partition = partitions
                .iter()
                .max_by_key(|p| p.partition_a.len().max(p.partition_b.len()))
                .map(|p| {
                    if p.partition_a.len() >= p.partition_b.len() {
                        &p.partition_a
                    } else {
                        &p.partition_b
                    }
                });

            if let Some(part) = primary_partition {
                if let Some(primary) = part.iter().min() {
                    return SplitBrainAction::ForcePrimary(primary.clone());
                }
            }
        }

        let total_views = self.views.len();
        if total_views > 0 && self.active_node_count() * 2 < total_views {
            return SplitBrainAction::Shutdown;
        }

        SplitBrainAction::WaitForHealing
    }

    /// Check whether a view from a node is stale given the current timestamp.
    pub fn is_stale(&self, node_id: &str, now_ms: u64) -> bool {
        self.views
            .get(node_id)
            .map(|v| now_ms.saturating_sub(v.last_updated) > self.partition_timeout_ms)
            .unwrap_or(false)
    }

    /// Remove a stale view for a given node.
    pub fn remove_view(&mut self, node_id: &str) {
        self.views.remove(node_id);
    }

    /// Return all known node IDs.
    pub fn known_nodes(&self) -> Vec<String> {
        let mut nodes: Vec<String> = self.views.keys().cloned().collect();
        nodes.sort();
        nodes
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn find_connected_groups<F>(
    node_ids: &[String],
    _views: &HashMap<String, ClusterView>,
    can_see: F,
) -> Vec<Vec<String>>
where
    F: Fn(&str, &str) -> bool,
{
    let n = node_ids.len();
    let mut component: Vec<usize> = (0..n).collect();

    // Union-find: merge nodes that can see each other
    for i in 0..n {
        for j in (i + 1)..n {
            let a = &node_ids[i];
            let b = &node_ids[j];
            if can_see(a, b) || can_see(b, a) {
                // Union i and j
                let ci = component[i];
                let cj = component[j];
                for c in component.iter_mut() {
                    if *c == cj {
                        *c = ci;
                    }
                }
            }
        }
    }

    // Collect components
    let mut groups: HashMap<usize, Vec<String>> = HashMap::new();
    for (i, &comp) in component.iter().enumerate() {
        groups.entry(comp).or_default().push(node_ids[i].clone());
    }

    let mut result: Vec<Vec<String>> = groups.into_values().collect();
    for group in &mut result {
        group.sort();
    }
    result.sort_by(|a, b| a[0].cmp(&b[0]));
    result
}

fn compute_confidence(a: &[String], b: &[String], total: usize) -> f64 {
    if total == 0 {
        return 0.0;
    }
    // Confidence is higher when both sides are larger relative to total
    let ratio_a = a.len() as f64 / total as f64;
    let ratio_b = b.len() as f64 / total as f64;
    (ratio_a + ratio_b).min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_active_view(node_id: &str, peers: &[&str], ts: u64) -> ClusterView {
        let mut view = ClusterView::new(node_id, ts);
        view.known_nodes
            .insert(node_id.to_string(), NodeState::Active);
        for &peer in peers {
            view.known_nodes.insert(peer.to_string(), NodeState::Active);
        }
        view
    }

    fn make_partitioned_view(
        node_id: &str,
        visible: &[&str],
        invisible: &[&str],
        ts: u64,
    ) -> ClusterView {
        let mut view = ClusterView::new(node_id, ts);
        view.known_nodes
            .insert(node_id.to_string(), NodeState::Active);
        for &peer in visible {
            view.known_nodes.insert(peer.to_string(), NodeState::Active);
        }
        for &peer in invisible {
            view.known_nodes
                .insert(peer.to_string(), NodeState::Partitioned);
        }
        view
    }

    #[test]
    fn test_new_detector() {
        let det = SplitBrainDetector::new(3, 5000);
        assert_eq!(det.active_node_count(), 0);
    }

    #[test]
    fn test_update_view() {
        let mut det = SplitBrainDetector::new(2, 5000);
        det.update_view(make_active_view("n1", &["n2"], 1000));
        assert_eq!(det.known_nodes().len(), 1);
    }

    #[test]
    fn test_known_nodes_sorted() {
        let mut det = SplitBrainDetector::new(2, 5000);
        det.update_view(make_active_view("n3", &["n1", "n2"], 1000));
        det.update_view(make_active_view("n1", &["n2", "n3"], 1000));
        det.update_view(make_active_view("n2", &["n1", "n3"], 1000));
        let nodes = det.known_nodes();
        assert_eq!(nodes, vec!["n1", "n2", "n3"]);
    }

    #[test]
    fn test_no_partition_healthy_cluster() {
        let mut det = SplitBrainDetector::new(2, 5000);
        det.update_view(make_active_view("n1", &["n2", "n3"], 1000));
        det.update_view(make_active_view("n2", &["n1", "n3"], 1000));
        det.update_view(make_active_view("n3", &["n1", "n2"], 1000));
        let parts = det.detect_partitions();
        assert!(parts.is_empty());
    }

    #[test]
    fn test_detect_partition_two_groups() {
        let mut det = SplitBrainDetector::new(2, 5000);
        // n1,n2 can see each other but not n3,n4
        det.update_view(make_partitioned_view("n1", &["n2"], &["n3", "n4"], 1000));
        det.update_view(make_partitioned_view("n2", &["n1"], &["n3", "n4"], 1000));
        det.update_view(make_partitioned_view("n3", &["n4"], &["n1", "n2"], 1000));
        det.update_view(make_partitioned_view("n4", &["n3"], &["n1", "n2"], 1000));
        let parts = det.detect_partitions();
        assert!(!parts.is_empty());
    }

    #[test]
    fn test_detect_partition_confidence_positive() {
        let mut det = SplitBrainDetector::new(2, 5000);
        det.update_view(make_partitioned_view("n1", &[], &["n2"], 1000));
        det.update_view(make_partitioned_view("n2", &[], &["n1"], 1000));
        let parts = det.detect_partitions();
        if !parts.is_empty() {
            assert!(parts[0].confidence > 0.0);
            assert!(parts[0].confidence <= 1.0);
        }
    }

    #[test]
    fn test_has_quorum_true() {
        let mut det = SplitBrainDetector::new(2, 5000);
        det.update_view(make_active_view("n1", &["n2"], 1000));
        det.update_view(make_active_view("n2", &["n1"], 1000));
        let nodes: Vec<String> = vec!["n1".to_string(), "n2".to_string()];
        assert!(det.has_quorum(&nodes));
    }

    #[test]
    fn test_has_quorum_false_insufficient() {
        let mut det = SplitBrainDetector::new(3, 5000);
        det.update_view(make_active_view("n1", &[], 1000));
        let nodes = vec!["n1".to_string()];
        assert!(!det.has_quorum(&nodes));
    }

    #[test]
    fn test_has_quorum_empty_set() {
        let det = SplitBrainDetector::new(1, 5000);
        assert!(!det.has_quorum(&[]));
    }

    #[test]
    fn test_recommend_no_action_healthy() {
        let mut det = SplitBrainDetector::new(2, 5000);
        det.update_view(make_active_view("n1", &["n2"], 1000));
        det.update_view(make_active_view("n2", &["n1"], 1000));
        assert_eq!(det.recommend_action(), SplitBrainAction::NoAction);
    }

    #[test]
    fn test_recommend_force_primary_on_partition() {
        let mut det = SplitBrainDetector::new(2, 5000);
        det.update_view(make_partitioned_view("n1", &["n2"], &["n3", "n4"], 1000));
        det.update_view(make_partitioned_view("n2", &["n1"], &["n3", "n4"], 1000));
        det.update_view(make_partitioned_view("n3", &["n4"], &["n1", "n2"], 1000));
        det.update_view(make_partitioned_view("n4", &["n3"], &["n1", "n2"], 1000));
        let action = det.recommend_action();
        // Should recommend ForcePrimary or WaitForHealing
        assert!(matches!(
            action,
            SplitBrainAction::ForcePrimary(_)
                | SplitBrainAction::WaitForHealing
                | SplitBrainAction::Shutdown
        ));
    }

    #[test]
    fn test_active_node_count_all_active() {
        let mut det = SplitBrainDetector::new(2, 5000);
        det.update_view(make_active_view("n1", &["n2"], 1000));
        det.update_view(make_active_view("n2", &["n1"], 1000));
        assert_eq!(det.active_node_count(), 2);
    }

    #[test]
    fn test_is_stale_true() {
        let mut det = SplitBrainDetector::new(2, 1000);
        det.update_view(make_active_view("n1", &[], 0));
        assert!(det.is_stale("n1", 2000));
    }

    #[test]
    fn test_is_stale_false() {
        let mut det = SplitBrainDetector::new(2, 5000);
        det.update_view(make_active_view("n1", &[], 1000));
        assert!(!det.is_stale("n1", 1500));
    }

    #[test]
    fn test_is_stale_unknown_node() {
        let det = SplitBrainDetector::new(2, 5000);
        assert!(!det.is_stale("unknown", 9999));
    }

    #[test]
    fn test_remove_view() {
        let mut det = SplitBrainDetector::new(2, 5000);
        det.update_view(make_active_view("n1", &[], 1000));
        det.remove_view("n1");
        assert!(det.known_nodes().is_empty());
    }

    #[test]
    fn test_single_node_no_partition() {
        let mut det = SplitBrainDetector::new(1, 5000);
        det.update_view(make_active_view("n1", &[], 1000));
        assert!(det.detect_partitions().is_empty());
    }

    #[test]
    fn test_update_view_overwrites() {
        let mut det = SplitBrainDetector::new(2, 5000);
        det.update_view(make_active_view("n1", &["n2"], 1000));
        det.update_view(make_active_view("n1", &["n2", "n3"], 2000));
        let view = det.views.get("n1").unwrap();
        assert_eq!(view.last_updated, 2000);
    }

    #[test]
    fn test_cluster_view_builder() {
        let view = ClusterView::new("n1", 500)
            .with_node("n2", NodeState::Active)
            .with_node("n3", NodeState::Unreachable);
        assert_eq!(view.known_nodes.len(), 2);
    }

    #[test]
    fn test_node_state_variants() {
        assert_ne!(NodeState::Active, NodeState::Suspected);
        assert_ne!(NodeState::Unreachable, NodeState::Partitioned);
    }

    #[test]
    fn test_split_brain_action_debug() {
        let a = SplitBrainAction::ForcePrimary("n1".to_string());
        let s = format!("{a:?}");
        assert!(s.contains("ForcePrimary"));
    }

    #[test]
    fn test_compute_confidence_zero_total() {
        assert_eq!(compute_confidence(&[], &[], 0), 0.0);
    }

    #[test]
    fn test_compute_confidence_full_coverage() {
        let a = vec!["n1".to_string(), "n2".to_string()];
        let b = vec!["n3".to_string(), "n4".to_string()];
        let conf = compute_confidence(&a, &b, 4);
        assert!((conf - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_partition_report_fields() {
        let report = PartitionReport {
            detected_at: 1000,
            partition_a: vec!["n1".to_string()],
            partition_b: vec!["n2".to_string()],
            confidence: 0.8,
        };
        assert_eq!(report.detected_at, 1000);
        assert_eq!(report.confidence, 0.8);
    }

    #[test]
    fn test_recommend_no_action_no_views() {
        let det = SplitBrainDetector::new(3, 5000);
        assert_eq!(det.recommend_action(), SplitBrainAction::NoAction);
    }

    #[test]
    fn test_three_node_partition_asymmetric() {
        let mut det = SplitBrainDetector::new(2, 5000);
        // n1 can see n2 but not n3; n3 is isolated
        det.update_view(make_partitioned_view("n1", &["n2"], &["n3"], 1000));
        det.update_view(make_partitioned_view("n2", &["n1"], &["n3"], 1000));
        det.update_view(make_partitioned_view("n3", &[], &["n1", "n2"], 1000));
        let parts = det.detect_partitions();
        assert!(!parts.is_empty());
    }
}
