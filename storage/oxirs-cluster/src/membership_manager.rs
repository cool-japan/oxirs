//! Cluster membership management.
//!
//! Tracks member roles, states, heartbeats, and failure detection for a distributed cluster.

use std::collections::HashMap;

/// Role of a cluster member.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemberRole {
    Leader,
    Follower,
    Candidate,
    Observer,
}

/// Lifecycle state of a cluster member.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemberState {
    Joining,
    Active,
    Leaving,
    Failed,
    Removed,
}

/// A member in the cluster.
#[derive(Debug, Clone)]
pub struct Member {
    pub id: String,
    pub address: String,
    pub role: MemberRole,
    pub state: MemberState,
    pub joined_at: u64,
    pub last_heartbeat: u64,
    pub metadata: HashMap<String, String>,
}

impl Member {
    /// Create a new active follower member.
    pub fn new(id: impl Into<String>, address: impl Into<String>, joined_at: u64) -> Self {
        Self {
            id: id.into(),
            address: address.into(),
            role: MemberRole::Follower,
            state: MemberState::Active,
            joined_at,
            last_heartbeat: joined_at,
            metadata: HashMap::new(),
        }
    }
}

/// Type of membership event.
#[derive(Debug, Clone, PartialEq)]
pub enum MembershipEventType {
    Joined,
    Left,
    FailureDetected,
    RoleChanged(MemberRole),
    StateChanged(MemberState),
}

/// An event recorded in the membership log.
#[derive(Debug, Clone)]
pub struct MembershipEvent {
    pub member_id: String,
    pub event_type: MembershipEventType,
    pub timestamp: u64,
}

impl MembershipEvent {
    fn new(
        member_id: impl Into<String>,
        event_type: MembershipEventType,
        timestamp: u64,
    ) -> Self {
        Self {
            member_id: member_id.into(),
            event_type,
            timestamp,
        }
    }
}

/// Manages cluster membership, heartbeats, and failure detection.
pub struct MembershipManager {
    local_id: String,
    members: HashMap<String, Member>,
    event_log: Vec<MembershipEvent>,
    heartbeat_timeout_ms: u64,
}

impl MembershipManager {
    /// Create a new membership manager with the local member pre-registered.
    pub fn new(local_id: String, address: String, heartbeat_timeout_ms: u64) -> Self {
        let mut members = HashMap::new();
        let now = 0u64; // local member joined at time 0 conceptually
        let local_member = Member {
            id: local_id.clone(),
            address,
            role: MemberRole::Follower,
            state: MemberState::Active,
            joined_at: now,
            last_heartbeat: now,
            metadata: HashMap::new(),
        };
        members.insert(local_id.clone(), local_member);

        Self {
            local_id,
            members,
            event_log: Vec::new(),
            heartbeat_timeout_ms,
        }
    }

    /// Add a new member to the cluster.
    pub fn add_member(&mut self, member: Member, current_time_ms: u64) {
        let event = MembershipEvent::new(
            member.id.clone(),
            MembershipEventType::Joined,
            current_time_ms,
        );
        self.members.insert(member.id.clone(), member);
        self.event_log.push(event);
    }

    /// Remove a member from the cluster by ID.
    ///
    /// Returns `true` if the member existed and was removed.
    pub fn remove_member(&mut self, id: &str, current_time_ms: u64) -> bool {
        if self.members.remove(id).is_some() {
            self.event_log.push(MembershipEvent::new(
                id,
                MembershipEventType::Left,
                current_time_ms,
            ));
            true
        } else {
            false
        }
    }

    /// Update the heartbeat timestamp for a member.
    ///
    /// Returns `true` if the member was found.
    pub fn update_heartbeat(&mut self, id: &str, current_time_ms: u64) -> bool {
        if let Some(member) = self.members.get_mut(id) {
            member.last_heartbeat = current_time_ms;
            true
        } else {
            false
        }
    }

    /// Set the role of a member.
    ///
    /// Returns `true` if the member was found.
    pub fn set_role(&mut self, id: &str, role: MemberRole, current_time_ms: u64) -> bool {
        if let Some(member) = self.members.get_mut(id) {
            member.role = role.clone();
            self.event_log.push(MembershipEvent::new(
                id,
                MembershipEventType::RoleChanged(role),
                current_time_ms,
            ));
            true
        } else {
            false
        }
    }

    /// Set the state of a member.
    ///
    /// Returns `true` if the member was found.
    pub fn set_state(&mut self, id: &str, state: MemberState, current_time_ms: u64) -> bool {
        if let Some(member) = self.members.get_mut(id) {
            member.state = state.clone();
            self.event_log.push(MembershipEvent::new(
                id,
                MembershipEventType::StateChanged(state),
                current_time_ms,
            ));
            true
        } else {
            false
        }
    }

    /// Detect members whose heartbeat has expired (missed heartbeat).
    ///
    /// Marks detected members as `MemberState::Failed` and returns their IDs.
    pub fn detect_failures(&mut self, current_time_ms: u64) -> Vec<String> {
        let timeout = self.heartbeat_timeout_ms;
        let local_id = self.local_id.clone();

        // Find failed members (excluding local node)
        let failed_ids: Vec<String> = self
            .members
            .iter()
            .filter(|(id, member)| {
                *id != &local_id
                    && member.state == MemberState::Active
                    && current_time_ms.saturating_sub(member.last_heartbeat) > timeout
            })
            .map(|(id, _)| id.clone())
            .collect();

        for id in &failed_ids {
            if let Some(member) = self.members.get_mut(id) {
                member.state = MemberState::Failed;
            }
            self.event_log.push(MembershipEvent::new(
                id.clone(),
                MembershipEventType::FailureDetected,
                current_time_ms,
            ));
        }

        failed_ids
    }

    /// Get all members.
    pub fn members(&self) -> Vec<&Member> {
        self.members.values().collect()
    }

    /// Get all active members.
    pub fn active_members(&self) -> Vec<&Member> {
        self.members
            .values()
            .filter(|m| m.state == MemberState::Active)
            .collect()
    }

    /// Get the current leader, if any.
    pub fn leader(&self) -> Option<&Member> {
        self.members
            .values()
            .find(|m| m.role == MemberRole::Leader && m.state == MemberState::Active)
    }

    /// Get total member count.
    pub fn member_count(&self) -> usize {
        self.members.len()
    }

    /// Get total event count.
    pub fn event_count(&self) -> usize {
        self.event_log.len()
    }

    /// Look up a member by ID.
    pub fn get_member(&self, id: &str) -> Option<&Member> {
        self.members.get(id)
    }

    /// Get the local node ID.
    pub fn local_id(&self) -> &str {
        &self.local_id
    }

    /// Get a reference to the event log.
    pub fn event_log(&self) -> &[MembershipEvent] {
        &self.event_log
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_manager() -> MembershipManager {
        MembershipManager::new("local".to_string(), "127.0.0.1:9000".to_string(), 5000)
    }

    fn make_member(id: &str, address: &str, time: u64) -> Member {
        Member::new(id, address, time)
    }

    #[test]
    fn test_new_creates_local_member() {
        let mgr = make_manager();
        assert_eq!(mgr.member_count(), 1);
        let m = mgr.get_member("local").unwrap();
        assert_eq!(m.id, "local");
    }

    #[test]
    fn test_add_member_increases_count() {
        let mut mgr = make_manager();
        mgr.add_member(make_member("node1", "192.168.1.1:9000", 100), 100);
        assert_eq!(mgr.member_count(), 2);
    }

    #[test]
    fn test_add_member_records_event() {
        let mut mgr = make_manager();
        mgr.add_member(make_member("node1", "192.168.1.1:9000", 100), 100);
        assert_eq!(mgr.event_count(), 1);
        let event = &mgr.event_log()[0];
        assert_eq!(event.member_id, "node1");
        assert_eq!(event.event_type, MembershipEventType::Joined);
    }

    #[test]
    fn test_remove_member_returns_true_when_found() {
        let mut mgr = make_manager();
        mgr.add_member(make_member("node1", "addr", 100), 100);
        assert!(mgr.remove_member("node1", 200));
        assert_eq!(mgr.member_count(), 1);
    }

    #[test]
    fn test_remove_member_returns_false_when_not_found() {
        let mut mgr = make_manager();
        assert!(!mgr.remove_member("nonexistent", 100));
    }

    #[test]
    fn test_remove_member_records_event() {
        let mut mgr = make_manager();
        mgr.add_member(make_member("node1", "addr", 100), 100);
        mgr.remove_member("node1", 200);
        let events: Vec<_> = mgr
            .event_log()
            .iter()
            .filter(|e| e.event_type == MembershipEventType::Left)
            .collect();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].member_id, "node1");
    }

    #[test]
    fn test_update_heartbeat_returns_true() {
        let mut mgr = make_manager();
        mgr.add_member(make_member("node1", "addr", 100), 100);
        assert!(mgr.update_heartbeat("node1", 500));
        let m = mgr.get_member("node1").unwrap();
        assert_eq!(m.last_heartbeat, 500);
    }

    #[test]
    fn test_update_heartbeat_returns_false_for_unknown() {
        let mut mgr = make_manager();
        assert!(!mgr.update_heartbeat("unknown", 100));
    }

    #[test]
    fn test_set_role_changes_role() {
        let mut mgr = make_manager();
        mgr.add_member(make_member("node1", "addr", 100), 100);
        assert!(mgr.set_role("node1", MemberRole::Leader, 200));
        let m = mgr.get_member("node1").unwrap();
        assert_eq!(m.role, MemberRole::Leader);
    }

    #[test]
    fn test_set_role_records_event() {
        let mut mgr = make_manager();
        mgr.add_member(make_member("node1", "addr", 100), 100);
        mgr.set_role("node1", MemberRole::Candidate, 200);
        let events: Vec<_> = mgr
            .event_log()
            .iter()
            .filter(|e| matches!(&e.event_type, MembershipEventType::RoleChanged(_)))
            .collect();
        assert_eq!(events.len(), 1);
    }

    #[test]
    fn test_set_role_returns_false_for_unknown() {
        let mut mgr = make_manager();
        assert!(!mgr.set_role("unknown", MemberRole::Leader, 100));
    }

    #[test]
    fn test_set_state_changes_state() {
        let mut mgr = make_manager();
        mgr.add_member(make_member("node1", "addr", 100), 100);
        assert!(mgr.set_state("node1", MemberState::Leaving, 200));
        let m = mgr.get_member("node1").unwrap();
        assert_eq!(m.state, MemberState::Leaving);
    }

    #[test]
    fn test_set_state_records_event() {
        let mut mgr = make_manager();
        mgr.add_member(make_member("node1", "addr", 100), 100);
        mgr.set_state("node1", MemberState::Leaving, 200);
        let events: Vec<_> = mgr
            .event_log()
            .iter()
            .filter(|e| matches!(&e.event_type, MembershipEventType::StateChanged(_)))
            .collect();
        assert_eq!(events.len(), 1);
    }

    #[test]
    fn test_set_state_returns_false_for_unknown() {
        let mut mgr = make_manager();
        assert!(!mgr.set_state("unknown", MemberState::Failed, 100));
    }

    #[test]
    fn test_detect_failures_marks_timed_out_members() {
        let mut mgr = make_manager();
        mgr.add_member(make_member("node1", "addr", 0), 0);
        // node1's last_heartbeat = 0; timeout = 5000; detect at 10000
        let failed = mgr.detect_failures(10_000);
        assert!(failed.contains(&"node1".to_string()));
        let m = mgr.get_member("node1").unwrap();
        assert_eq!(m.state, MemberState::Failed);
    }

    #[test]
    fn test_detect_failures_skips_recent_heartbeat() {
        let mut mgr = make_manager();
        mgr.add_member(make_member("node1", "addr", 0), 0);
        mgr.update_heartbeat("node1", 9500);
        // detect at 10000: elapsed = 500 < 5000 timeout
        let failed = mgr.detect_failures(10_000);
        assert!(failed.is_empty());
    }

    #[test]
    fn test_detect_failures_skips_local_node() {
        let mut mgr = MembershipManager::new(
            "local".to_string(),
            "127.0.0.1:9000".to_string(),
            1000,
        );
        // local heartbeat is 0; detect at 100_000 — should not flag local
        let failed = mgr.detect_failures(100_000);
        assert!(!failed.contains(&"local".to_string()));
    }

    #[test]
    fn test_detect_failures_records_event() {
        let mut mgr = make_manager();
        mgr.add_member(make_member("node1", "addr", 0), 0);
        mgr.detect_failures(10_000);
        let events: Vec<_> = mgr
            .event_log()
            .iter()
            .filter(|e| e.event_type == MembershipEventType::FailureDetected)
            .collect();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].member_id, "node1");
    }

    #[test]
    fn test_detect_failures_skips_already_failed() {
        let mut mgr = make_manager();
        let mut m = make_member("node1", "addr", 0);
        m.state = MemberState::Failed;
        mgr.add_member(m, 0);
        let failed = mgr.detect_failures(10_000);
        assert!(failed.is_empty());
    }

    #[test]
    fn test_leader_returns_active_leader() {
        let mut mgr = make_manager();
        mgr.add_member(make_member("node1", "addr", 100), 100);
        mgr.set_role("node1", MemberRole::Leader, 200);
        let leader = mgr.leader();
        assert!(leader.is_some());
        assert_eq!(leader.unwrap().id, "node1");
    }

    #[test]
    fn test_leader_returns_none_when_no_leader() {
        let mgr = make_manager();
        assert!(mgr.leader().is_none());
    }

    #[test]
    fn test_leader_returns_none_when_leader_failed() {
        let mut mgr = make_manager();
        mgr.add_member(make_member("node1", "addr", 0), 0);
        mgr.set_role("node1", MemberRole::Leader, 100);
        mgr.set_state("node1", MemberState::Failed, 200);
        assert!(mgr.leader().is_none());
    }

    #[test]
    fn test_active_members_filter() {
        let mut mgr = make_manager();
        mgr.add_member(make_member("node1", "addr", 100), 100);
        mgr.add_member(make_member("node2", "addr2", 100), 100);
        mgr.set_state("node1", MemberState::Failed, 200);
        let active = mgr.active_members();
        // local + node2 = 2
        assert_eq!(active.len(), 2);
        assert!(active.iter().all(|m| m.state == MemberState::Active));
    }

    #[test]
    fn test_member_count_after_operations() {
        let mut mgr = make_manager();
        assert_eq!(mgr.member_count(), 1);
        mgr.add_member(make_member("n1", "a", 0), 0);
        mgr.add_member(make_member("n2", "b", 0), 0);
        assert_eq!(mgr.member_count(), 3);
        mgr.remove_member("n1", 100);
        assert_eq!(mgr.member_count(), 2);
    }

    #[test]
    fn test_event_count_grows() {
        let mut mgr = make_manager();
        assert_eq!(mgr.event_count(), 0);
        mgr.add_member(make_member("n1", "a", 0), 0);
        assert_eq!(mgr.event_count(), 1);
        mgr.set_role("n1", MemberRole::Candidate, 100);
        assert_eq!(mgr.event_count(), 2);
    }

    #[test]
    fn test_get_member_returns_none_for_unknown() {
        let mgr = make_manager();
        assert!(mgr.get_member("unknown").is_none());
    }

    #[test]
    fn test_members_list_contains_all() {
        let mut mgr = make_manager();
        mgr.add_member(make_member("n1", "a", 0), 0);
        mgr.add_member(make_member("n2", "b", 0), 0);
        let all = mgr.members();
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn test_detect_multiple_failures() {
        let mut mgr = make_manager();
        mgr.add_member(make_member("n1", "a", 0), 0);
        mgr.add_member(make_member("n2", "b", 0), 0);
        let failed = mgr.detect_failures(10_000);
        assert_eq!(failed.len(), 2);
    }

    #[test]
    fn test_metadata_stored_in_member() {
        let mut mgr = make_manager();
        let mut m = make_member("n1", "a", 0);
        m.metadata.insert("region".to_string(), "us-east".to_string());
        mgr.add_member(m, 0);
        let retrieved = mgr.get_member("n1").unwrap();
        assert_eq!(retrieved.metadata.get("region"), Some(&"us-east".to_string()));
    }
}
