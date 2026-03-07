//! Simplified in-memory event sourcing utilities (v1.1.0).
//!
//! Provides `SimpleEventStore`, `SimpleSnapshotStore`, `SimpleEventBus`,
//! `ProjectionRunner`, and `EventStreamIter` for lightweight event sourcing
//! without the async/persistence overhead of the full `EventStore`.

use std::collections::HashMap;

/// A simple domain event for in-memory append-only log.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SimpleEvent {
    /// Unique event identifier (auto-incremented).
    pub id: u64,
    /// ID of the aggregate this event belongs to.
    pub aggregate_id: String,
    /// Type label for the event (e.g. `"OrderPlaced"`).
    pub event_type: String,
    /// JSON or text payload.
    pub payload: String,
    /// Monotonically increasing version within the aggregate.
    pub version: u64,
    /// Unix timestamp in seconds.
    pub timestamp: u64,
}

/// Append-only, in-memory simple event store.
pub struct SimpleEventStore {
    events: Vec<SimpleEvent>,
    next_id: u64,
    versions: HashMap<String, u64>,
}

impl SimpleEventStore {
    /// Create an empty event store.
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            next_id: 1,
            versions: HashMap::new(),
        }
    }

    /// Append a new event for `aggregate_id`.
    pub fn append(
        &mut self,
        aggregate_id: impl Into<String>,
        event_type: impl Into<String>,
        payload: impl Into<String>,
    ) -> SimpleEvent {
        use std::time::{SystemTime, UNIX_EPOCH};
        let aggregate_id = aggregate_id.into();
        let version = self.versions.entry(aggregate_id.clone()).or_insert(0);
        *version += 1;
        let event = SimpleEvent {
            id: self.next_id,
            aggregate_id: aggregate_id.clone(),
            event_type: event_type.into(),
            payload: payload.into(),
            version: *version,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        };
        self.next_id += 1;
        self.events.push(event.clone());
        event
    }

    /// Load all events for the given aggregate, ordered by version.
    pub fn load_aggregate(&self, aggregate_id: &str) -> Vec<SimpleEvent> {
        self.events
            .iter()
            .filter(|e| e.aggregate_id == aggregate_id)
            .cloned()
            .collect()
    }

    /// Load events for `aggregate_id` starting at `from_version` (inclusive).
    pub fn load_from_version(&self, aggregate_id: &str, from_version: u64) -> Vec<SimpleEvent> {
        self.events
            .iter()
            .filter(|e| e.aggregate_id == aggregate_id && e.version >= from_version)
            .cloned()
            .collect()
    }

    /// Load every event across all aggregates.
    pub fn load_all_events(&self) -> Vec<SimpleEvent> {
        self.events.clone()
    }

    /// Total number of events stored.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// True when the store is empty.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Current version for `aggregate_id`.
    pub fn current_version(&self, aggregate_id: &str) -> u64 {
        self.versions.get(aggregate_id).copied().unwrap_or(0)
    }
}

impl Default for SimpleEventStore {
    fn default() -> Self {
        Self::new()
    }
}

/// An iterator over simple events with optional filtering.
pub struct EventStreamIter {
    events: Vec<SimpleEvent>,
    position: usize,
    filter_aggregate: Option<String>,
    filter_type: Option<String>,
}

impl EventStreamIter {
    /// Create a stream over a list of events.
    pub fn new(events: Vec<SimpleEvent>) -> Self {
        Self {
            events,
            position: 0,
            filter_aggregate: None,
            filter_type: None,
        }
    }

    /// Filter events to a single aggregate.
    pub fn for_aggregate(mut self, aggregate_id: impl Into<String>) -> Self {
        self.filter_aggregate = Some(aggregate_id.into());
        self
    }

    /// Filter events to a single event type.
    pub fn for_type(mut self, event_type: impl Into<String>) -> Self {
        self.filter_type = Some(event_type.into());
        self
    }

    fn matches(&self, event: &SimpleEvent) -> bool {
        if let Some(ref agg) = self.filter_aggregate {
            if &event.aggregate_id != agg {
                return false;
            }
        }
        if let Some(ref et) = self.filter_type {
            if &event.event_type != et {
                return false;
            }
        }
        true
    }
}

impl Iterator for EventStreamIter {
    type Item = SimpleEvent;

    fn next(&mut self) -> Option<Self::Item> {
        while self.position < self.events.len() {
            let ev = &self.events[self.position];
            self.position += 1;
            if self.matches(ev) {
                return Some(ev.clone());
            }
        }
        None
    }
}

/// A serialized snapshot of aggregate state at a given version.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SimpleSnapshot {
    /// Aggregate this snapshot belongs to.
    pub aggregate_id: String,
    /// Serialized state.
    pub state: String,
    /// Aggregate version at the time of the snapshot.
    pub version: u64,
}

/// In-memory store for aggregate snapshots.
pub struct SimpleSnapshotStore {
    snapshots: HashMap<String, SimpleSnapshot>,
}

impl SimpleSnapshotStore {
    /// Create an empty snapshot store.
    pub fn new() -> Self {
        Self {
            snapshots: HashMap::new(),
        }
    }

    /// Save or overwrite the snapshot for an aggregate.
    pub fn save(&mut self, snapshot: SimpleSnapshot) {
        self.snapshots
            .insert(snapshot.aggregate_id.clone(), snapshot);
    }

    /// Retrieve the latest snapshot for `aggregate_id`, if any.
    pub fn load_snapshot(&self, aggregate_id: &str) -> Option<&SimpleSnapshot> {
        self.snapshots.get(aggregate_id)
    }

    /// Remove the snapshot for `aggregate_id`.
    pub fn delete(&mut self, aggregate_id: &str) -> bool {
        self.snapshots.remove(aggregate_id).is_some()
    }

    /// Number of stored snapshots.
    pub fn len(&self) -> usize {
        self.snapshots.len()
    }

    /// True when no snapshots are stored.
    pub fn is_empty(&self) -> bool {
        self.snapshots.is_empty()
    }
}

impl Default for SimpleSnapshotStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Handler function type for the simple event bus.
pub type SimpleEventHandler = std::sync::Arc<dyn Fn(&SimpleEvent) + Send + Sync>;

/// A simple pub/sub event bus.
pub struct SimpleEventBus {
    subscriptions: HashMap<String, Vec<SimpleEventHandler>>,
    wildcard: Vec<SimpleEventHandler>,
}

impl SimpleEventBus {
    /// Create an empty event bus.
    pub fn new() -> Self {
        Self {
            subscriptions: HashMap::new(),
            wildcard: Vec::new(),
        }
    }

    /// Subscribe a handler to a specific `event_type`. Use `"*"` for all events.
    pub fn subscribe(&mut self, event_type: impl Into<String>, handler: SimpleEventHandler) {
        let key = event_type.into();
        if key == "*" {
            self.wildcard.push(handler);
        } else {
            self.subscriptions.entry(key).or_default().push(handler);
        }
    }

    /// Publish an event, invoking all matching handlers synchronously.
    pub fn publish(&self, event: &SimpleEvent) {
        if let Some(handlers) = self.subscriptions.get(&event.event_type) {
            for handler in handlers {
                handler(event);
            }
        }
        for handler in &self.wildcard {
            handler(event);
        }
    }

    /// Total number of type-specific subscriptions.
    pub fn subscription_count(&self) -> usize {
        self.subscriptions.values().map(|v| v.len()).sum()
    }

    /// Number of wildcard subscriptions.
    pub fn wildcard_count(&self) -> usize {
        self.wildcard.len()
    }
}

impl Default for SimpleEventBus {
    fn default() -> Self {
        Self::new()
    }
}

/// Applies events to build a read model (projection).
pub struct ProjectionRunner {
    /// Human-readable name for the projection.
    pub name: String,
    /// Number of events processed so far.
    processed: u64,
}

impl ProjectionRunner {
    /// Create a named projection runner.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            processed: 0,
        }
    }

    /// Apply a handler function over every event in `store`, returning the final state.
    pub fn run<S, F>(&mut self, store: &SimpleEventStore, initial: S, mut handler: F) -> S
    where
        F: FnMut(S, &SimpleEvent) -> S,
    {
        let mut state = initial;
        for event in store.load_all_events() {
            state = handler(state, &event);
            self.processed += 1;
        }
        state
    }

    /// Apply a handler over events for a single aggregate.
    pub fn run_for_aggregate<S, F>(
        &mut self,
        store: &SimpleEventStore,
        aggregate_id: &str,
        initial: S,
        mut handler: F,
    ) -> S
    where
        F: FnMut(S, &SimpleEvent) -> S,
    {
        let mut state = initial;
        for event in store.load_aggregate(aggregate_id) {
            state = handler(state, &event);
            self.processed += 1;
        }
        state
    }

    /// Events processed so far.
    pub fn processed_count(&self) -> u64 {
        self.processed
    }
}
