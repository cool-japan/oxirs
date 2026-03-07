//! GraphQL subscription lifecycle management.
//!
//! Provides in-memory subscription tracking, event delivery, and
//! lifecycle management for GraphQL subscriptions without requiring
//! an async runtime.

use std::collections::HashMap;
use std::fmt;

/// The current status of a GraphQL subscription.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubscriptionStatus {
    /// Subscription is active and accepting events.
    Active,
    /// Subscription is temporarily paused; events are not delivered.
    Paused,
    /// Subscription has completed normally.
    Completed,
    /// Subscription encountered an error.
    Error(String),
}

/// A filter applied to incoming events to decide whether to deliver them to a subscription.
#[derive(Debug, Clone, Default)]
pub struct EventFilter {
    /// If set, only events matching this operation name are delivered.
    pub operation_name: Option<String>,
    /// Key-value pairs that must all be present and equal in the event variables.
    pub variables_match: HashMap<String, String>,
}

impl EventFilter {
    /// Create a new filter that matches all events.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a filter that requires a specific operation name.
    pub fn with_operation(op_name: impl Into<String>) -> Self {
        Self {
            operation_name: Some(op_name.into()),
            variables_match: HashMap::new(),
        }
    }

    /// Add a variable match requirement.
    pub fn require_variable(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.variables_match.insert(key.into(), value.into());
        self
    }

    /// Check whether an event with the given operation name and variables passes this filter.
    pub fn matches(&self, op_name: &str, vars: &HashMap<String, String>) -> bool {
        // Check operation name if specified
        if let Some(required_op) = &self.operation_name {
            if required_op != op_name {
                return false;
            }
        }
        // All required variable key-value pairs must be present
        for (k, v) in &self.variables_match {
            match vars.get(k) {
                Some(val) if val == v => {}
                _ => return false,
            }
        }
        true
    }
}

/// A registered GraphQL subscription.
#[derive(Debug, Clone)]
pub struct Subscription {
    /// Unique identifier for this subscription.
    pub id: String,
    /// The GraphQL operation (subscription query text).
    pub operation: String,
    /// Variables provided for this subscription.
    pub variables: HashMap<String, String>,
    /// Filter controlling which events are delivered.
    pub filter: EventFilter,
    /// Current lifecycle status.
    pub status: SubscriptionStatus,
    /// Monotonic creation timestamp.
    pub created_at_ms: u64,
    /// Monotonic timestamp of the last delivered event (0 if none).
    pub last_event_ms: u64,
    /// Total number of events delivered to this subscription.
    pub event_count: u64,
    /// Sequence counter for events (incremented per delivered event).
    pub(crate) next_sequence: u64,
}

/// A single event delivered to a subscription.
#[derive(Debug, Clone)]
pub struct SubscriptionEvent {
    /// The id of the subscription receiving this event.
    pub subscription_id: String,
    /// Monotonically increasing sequence number per subscription.
    pub sequence: u64,
    /// The event payload (typically serialised GraphQL data).
    pub payload: String,
    /// Monotonic timestamp when this event was created.
    pub timestamp_ms: u64,
}

/// Error type for subscription manager operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubError {
    /// A subscription with this id already exists.
    AlreadyExists(String),
    /// No subscription was found with this id.
    NotFound(String),
}

impl fmt::Display for SubError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SubError::AlreadyExists(id) => write!(f, "Subscription already exists: {id}"),
            SubError::NotFound(id) => write!(f, "Subscription not found: {id}"),
        }
    }
}

impl std::error::Error for SubError {}

/// Aggregated statistics for the subscription manager.
#[derive(Debug, Clone)]
pub struct SubStats {
    /// Total number of registered subscriptions (all statuses).
    pub total: usize,
    /// Number of subscriptions with `Active` status.
    pub active: usize,
    /// Number of subscriptions with `Paused` status.
    pub paused: usize,
    /// Number of subscriptions with `Completed` status.
    pub completed: usize,
    /// Number of subscriptions with `Error` status.
    pub error: usize,
}

/// Manager for GraphQL subscription lifecycle and event delivery.
pub struct SubscriptionManager {
    subscriptions: HashMap<String, Subscription>,
    /// Global monotonic clock for timestamps.
    clock: u64,
}

impl SubscriptionManager {
    /// Create a new empty subscription manager.
    pub fn new() -> Self {
        Self {
            subscriptions: HashMap::new(),
            clock: 0,
        }
    }

    fn tick(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    /// Register a new subscription.
    ///
    /// Returns `Err(SubError::AlreadyExists)` if a subscription with the same id is already registered.
    pub fn subscribe(
        &mut self,
        id: impl Into<String>,
        operation: impl Into<String>,
        variables: HashMap<String, String>,
        filter: EventFilter,
    ) -> Result<(), SubError> {
        let id = id.into();
        if self.subscriptions.contains_key(&id) {
            return Err(SubError::AlreadyExists(id));
        }
        let ts = self.tick();
        let sub = Subscription {
            id: id.clone(),
            operation: operation.into(),
            variables,
            filter,
            status: SubscriptionStatus::Active,
            created_at_ms: ts,
            last_event_ms: 0,
            event_count: 0,
            next_sequence: 1,
        };
        self.subscriptions.insert(id, sub);
        Ok(())
    }

    /// Remove a subscription entirely.
    ///
    /// Returns `Err(SubError::NotFound)` if no such subscription exists.
    pub fn unsubscribe(&mut self, id: &str) -> Result<(), SubError> {
        self.subscriptions
            .remove(id)
            .ok_or_else(|| SubError::NotFound(id.to_string()))?;
        Ok(())
    }

    /// Pause a subscription so it no longer receives events.
    pub fn pause(&mut self, id: &str) -> Result<(), SubError> {
        let sub = self
            .subscriptions
            .get_mut(id)
            .ok_or_else(|| SubError::NotFound(id.to_string()))?;
        sub.status = SubscriptionStatus::Paused;
        Ok(())
    }

    /// Resume a previously paused subscription.
    pub fn resume(&mut self, id: &str) -> Result<(), SubError> {
        let sub = self
            .subscriptions
            .get_mut(id)
            .ok_or_else(|| SubError::NotFound(id.to_string()))?;
        sub.status = SubscriptionStatus::Active;
        Ok(())
    }

    /// Mark a subscription as completed.
    pub fn complete(&mut self, id: &str) -> Result<(), SubError> {
        let sub = self
            .subscriptions
            .get_mut(id)
            .ok_or_else(|| SubError::NotFound(id.to_string()))?;
        sub.status = SubscriptionStatus::Completed;
        Ok(())
    }

    /// Mark a subscription as errored.
    pub fn set_error(&mut self, id: &str, message: impl Into<String>) -> Result<(), SubError> {
        let sub = self
            .subscriptions
            .get_mut(id)
            .ok_or_else(|| SubError::NotFound(id.to_string()))?;
        sub.status = SubscriptionStatus::Error(message.into());
        Ok(())
    }

    /// Deliver `payload` to all `Active` subscriptions whose filter matches.
    ///
    /// Returns the list of events that were created and delivered.
    pub fn publish_event(
        &mut self,
        payload: &str,
        op_name: &str,
        vars: &HashMap<String, String>,
    ) -> Vec<SubscriptionEvent> {
        let ts = self.tick();
        let mut events = Vec::new();

        for sub in self.subscriptions.values_mut() {
            if sub.status != SubscriptionStatus::Active {
                continue;
            }
            if !sub.filter.matches(op_name, vars) {
                continue;
            }
            let seq = sub.next_sequence;
            sub.next_sequence += 1;
            sub.event_count += 1;
            sub.last_event_ms = ts;
            events.push(SubscriptionEvent {
                subscription_id: sub.id.clone(),
                sequence: seq,
                payload: payload.to_string(),
                timestamp_ms: ts,
            });
        }

        events
    }

    /// Retrieve a subscription by id.
    pub fn get(&self, id: &str) -> Option<&Subscription> {
        self.subscriptions.get(id)
    }

    /// Return all currently active subscriptions.
    pub fn active_subscriptions(&self) -> Vec<&Subscription> {
        self.subscriptions
            .values()
            .filter(|s| s.status == SubscriptionStatus::Active)
            .collect()
    }

    /// Remove all `Completed` subscriptions.
    pub fn cleanup_completed(&mut self) {
        self.subscriptions
            .retain(|_, s| s.status != SubscriptionStatus::Completed);
    }

    /// Compute aggregate statistics.
    pub fn stats(&self) -> SubStats {
        let mut active = 0;
        let mut paused = 0;
        let mut completed = 0;
        let mut error = 0;
        for sub in self.subscriptions.values() {
            match &sub.status {
                SubscriptionStatus::Active => active += 1,
                SubscriptionStatus::Paused => paused += 1,
                SubscriptionStatus::Completed => completed += 1,
                SubscriptionStatus::Error(_) => error += 1,
            }
        }
        SubStats {
            total: self.subscriptions.len(),
            active,
            paused,
            completed,
            error,
        }
    }

    /// Return the number of registered subscriptions.
    pub fn len(&self) -> usize {
        self.subscriptions.len()
    }

    /// Return `true` if there are no registered subscriptions.
    pub fn is_empty(&self) -> bool {
        self.subscriptions.is_empty()
    }
}

impl Default for SubscriptionManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_vars() -> HashMap<String, String> {
        HashMap::new()
    }

    fn vars(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    fn mgr() -> SubscriptionManager {
        SubscriptionManager::new()
    }

    // ── subscribe / get ───────────────────────────────────────────────────────

    #[test]
    fn test_subscribe_and_get() {
        let mut m = mgr();
        m.subscribe(
            "s1",
            "subscription Foo {}",
            empty_vars(),
            EventFilter::new(),
        )
        .unwrap();
        let sub = m.get("s1").unwrap();
        assert_eq!(sub.id, "s1");
        assert_eq!(sub.status, SubscriptionStatus::Active);
        assert_eq!(sub.event_count, 0);
    }

    #[test]
    fn test_subscribe_duplicate_error() {
        let mut m = mgr();
        m.subscribe("s1", "op", empty_vars(), EventFilter::new())
            .unwrap();
        let r = m.subscribe("s1", "op", empty_vars(), EventFilter::new());
        assert_eq!(r, Err(SubError::AlreadyExists("s1".to_string())));
    }

    #[test]
    fn test_get_nonexistent() {
        let m = mgr();
        assert!(m.get("nope").is_none());
    }

    // ── unsubscribe ───────────────────────────────────────────────────────────

    #[test]
    fn test_unsubscribe() {
        let mut m = mgr();
        m.subscribe("s1", "op", empty_vars(), EventFilter::new())
            .unwrap();
        m.unsubscribe("s1").unwrap();
        assert!(m.get("s1").is_none());
    }

    #[test]
    fn test_unsubscribe_not_found() {
        let mut m = mgr();
        let r = m.unsubscribe("ghost");
        assert_eq!(r, Err(SubError::NotFound("ghost".to_string())));
    }

    // ── pause / resume ────────────────────────────────────────────────────────

    #[test]
    fn test_pause_and_resume() {
        let mut m = mgr();
        m.subscribe("s1", "op", empty_vars(), EventFilter::new())
            .unwrap();
        m.pause("s1").unwrap();
        assert_eq!(m.get("s1").unwrap().status, SubscriptionStatus::Paused);
        m.resume("s1").unwrap();
        assert_eq!(m.get("s1").unwrap().status, SubscriptionStatus::Active);
    }

    #[test]
    fn test_pause_not_found() {
        let mut m = mgr();
        assert_eq!(m.pause("x"), Err(SubError::NotFound("x".to_string())));
    }

    #[test]
    fn test_resume_not_found() {
        let mut m = mgr();
        assert_eq!(m.resume("x"), Err(SubError::NotFound("x".to_string())));
    }

    // ── complete ──────────────────────────────────────────────────────────────

    #[test]
    fn test_complete() {
        let mut m = mgr();
        m.subscribe("s1", "op", empty_vars(), EventFilter::new())
            .unwrap();
        m.complete("s1").unwrap();
        assert_eq!(m.get("s1").unwrap().status, SubscriptionStatus::Completed);
    }

    #[test]
    fn test_complete_not_found() {
        let mut m = mgr();
        assert_eq!(m.complete("x"), Err(SubError::NotFound("x".to_string())));
    }

    // ── publish_event ─────────────────────────────────────────────────────────

    #[test]
    fn test_publish_delivers_to_active() {
        let mut m = mgr();
        m.subscribe("s1", "op", empty_vars(), EventFilter::new())
            .unwrap();
        let events = m.publish_event("{\"data\":1}", "op", &empty_vars());
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].subscription_id, "s1");
        assert_eq!(events[0].payload, "{\"data\":1}");
        assert_eq!(events[0].sequence, 1);
    }

    #[test]
    fn test_publish_does_not_deliver_to_paused() {
        let mut m = mgr();
        m.subscribe("s1", "op", empty_vars(), EventFilter::new())
            .unwrap();
        m.pause("s1").unwrap();
        let events = m.publish_event("payload", "op", &empty_vars());
        assert!(events.is_empty());
    }

    #[test]
    fn test_publish_does_not_deliver_to_completed() {
        let mut m = mgr();
        m.subscribe("s1", "op", empty_vars(), EventFilter::new())
            .unwrap();
        m.complete("s1").unwrap();
        let events = m.publish_event("payload", "op", &empty_vars());
        assert!(events.is_empty());
    }

    #[test]
    fn test_publish_increments_event_count() {
        let mut m = mgr();
        m.subscribe("s1", "op", empty_vars(), EventFilter::new())
            .unwrap();
        m.publish_event("e1", "op", &empty_vars());
        m.publish_event("e2", "op", &empty_vars());
        assert_eq!(m.get("s1").unwrap().event_count, 2);
    }

    #[test]
    fn test_publish_sequence_increments() {
        let mut m = mgr();
        m.subscribe("s1", "op", empty_vars(), EventFilter::new())
            .unwrap();
        let e1 = m.publish_event("e1", "op", &empty_vars());
        let e2 = m.publish_event("e2", "op", &empty_vars());
        assert_eq!(e1[0].sequence, 1);
        assert_eq!(e2[0].sequence, 2);
    }

    #[test]
    fn test_publish_multiple_subscribers() {
        let mut m = mgr();
        m.subscribe("s1", "op", empty_vars(), EventFilter::new())
            .unwrap();
        m.subscribe("s2", "op", empty_vars(), EventFilter::new())
            .unwrap();
        let events = m.publish_event("payload", "op", &empty_vars());
        assert_eq!(events.len(), 2);
    }

    // ── EventFilter ───────────────────────────────────────────────────────────

    #[test]
    fn test_filter_by_operation() {
        let mut m = mgr();
        m.subscribe(
            "s1",
            "op",
            empty_vars(),
            EventFilter::with_operation("targetOp"),
        )
        .unwrap();
        // Non-matching op
        let e1 = m.publish_event("p", "otherOp", &empty_vars());
        assert!(e1.is_empty());
        // Matching op
        let e2 = m.publish_event("p", "targetOp", &empty_vars());
        assert_eq!(e2.len(), 1);
    }

    #[test]
    fn test_filter_by_variable() {
        let mut m = mgr();
        let filter = EventFilter::new().require_variable("userId", "42");
        m.subscribe("s1", "op", empty_vars(), filter).unwrap();
        // No variables
        let e1 = m.publish_event("p", "op", &empty_vars());
        assert!(e1.is_empty());
        // Wrong value
        let e2 = m.publish_event("p", "op", &vars(&[("userId", "99")]));
        assert!(e2.is_empty());
        // Correct value
        let e3 = m.publish_event("p", "op", &vars(&[("userId", "42")]));
        assert_eq!(e3.len(), 1);
    }

    #[test]
    fn test_filter_matches_any_op_when_none_set() {
        let filter = EventFilter::new();
        assert!(filter.matches("anything", &empty_vars()));
    }

    // ── active_subscriptions ──────────────────────────────────────────────────

    #[test]
    fn test_active_subscriptions() {
        let mut m = mgr();
        m.subscribe("s1", "op", empty_vars(), EventFilter::new())
            .unwrap();
        m.subscribe("s2", "op", empty_vars(), EventFilter::new())
            .unwrap();
        m.subscribe("s3", "op", empty_vars(), EventFilter::new())
            .unwrap();
        m.pause("s2").unwrap();
        m.complete("s3").unwrap();
        let active = m.active_subscriptions();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].id, "s1");
    }

    // ── cleanup_completed ─────────────────────────────────────────────────────

    #[test]
    fn test_cleanup_completed() {
        let mut m = mgr();
        m.subscribe("s1", "op", empty_vars(), EventFilter::new())
            .unwrap();
        m.subscribe("s2", "op", empty_vars(), EventFilter::new())
            .unwrap();
        m.complete("s1").unwrap();
        m.cleanup_completed();
        assert!(m.get("s1").is_none());
        assert!(m.get("s2").is_some());
    }

    // ── stats ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_stats_initial() {
        let m = mgr();
        let s = m.stats();
        assert_eq!(s.total, 0);
        assert_eq!(s.active, 0);
    }

    #[test]
    fn test_stats_mixed() {
        let mut m = mgr();
        m.subscribe("s1", "op", empty_vars(), EventFilter::new())
            .unwrap();
        m.subscribe("s2", "op", empty_vars(), EventFilter::new())
            .unwrap();
        m.subscribe("s3", "op", empty_vars(), EventFilter::new())
            .unwrap();
        m.subscribe("s4", "op", empty_vars(), EventFilter::new())
            .unwrap();
        m.pause("s2").unwrap();
        m.complete("s3").unwrap();
        m.set_error("s4", "boom").unwrap();
        let s = m.stats();
        assert_eq!(s.total, 4);
        assert_eq!(s.active, 1);
        assert_eq!(s.paused, 1);
        assert_eq!(s.completed, 1);
        assert_eq!(s.error, 1);
    }

    // ── SubError display ──────────────────────────────────────────────────────

    #[test]
    fn test_sub_error_display() {
        let e1 = SubError::AlreadyExists("s1".to_string());
        assert!(e1.to_string().contains("s1"));
        let e2 = SubError::NotFound("s2".to_string());
        assert!(e2.to_string().contains("s2"));
    }

    // ── SubscriptionStatus ────────────────────────────────────────────────────

    #[test]
    fn test_subscription_status_clone_eq() {
        let s = SubscriptionStatus::Error("msg".to_string());
        let s2 = s.clone();
        assert_eq!(s, s2);
    }

    // ── default impl ──────────────────────────────────────────────────────────

    #[test]
    fn test_default() {
        let m = SubscriptionManager::default();
        assert!(m.is_empty());
    }
}
