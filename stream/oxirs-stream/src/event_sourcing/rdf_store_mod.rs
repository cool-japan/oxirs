//! RDF-specific event sourcing for stream changes.
//!
//! Lightweight store, snapshot, and projector for tracking triple-level and
//! graph-level changes.

use std::collections::{HashMap, HashSet};

/// Type of RDF stream event.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum EventType {
    /// A triple was added to the stream / dataset.
    TripleAdded,
    /// A triple was removed from the stream / dataset.
    TripleRemoved,
    /// A named graph was created.
    GraphCreated,
    /// A named graph was dropped.
    GraphDropped,
    /// Two datasets were merged.
    DatasetMerged,
}

impl std::fmt::Display for EventType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EventType::TripleAdded => write!(f, "TripleAdded"),
            EventType::TripleRemoved => write!(f, "TripleRemoved"),
            EventType::GraphCreated => write!(f, "GraphCreated"),
            EventType::GraphDropped => write!(f, "GraphDropped"),
            EventType::DatasetMerged => write!(f, "DatasetMerged"),
        }
    }
}

/// A single RDF stream event with all relevant context.
#[derive(Clone, Debug)]
pub struct RdfStreamEvent {
    /// Auto-assigned monotonic identifier.
    pub id: u64,
    /// Classification of the event.
    pub event_type: EventType,
    /// Subject IRI or blank node (for triple events).
    pub subject: Option<String>,
    /// Predicate IRI (for triple events).
    pub predicate: Option<String>,
    /// Object IRI, blank node, or literal (for triple events).
    pub object: Option<String>,
    /// Named graph IRI (for graph events or scoped triple events).
    pub graph: Option<String>,
    /// Unix timestamp in milliseconds when the event was appended.
    pub timestamp_ms: u64,
    /// Arbitrary user-supplied key-value metadata.
    pub metadata: HashMap<String, String>,
}

/// Returns the current time as milliseconds since UNIX epoch.
fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Append-only in-memory event store for RDF stream events.
#[derive(Debug, Default)]
pub struct RdfEventStore {
    events: Vec<RdfStreamEvent>,
    counter: u64,
}

impl RdfEventStore {
    /// Create an empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a new event and return its assigned id.
    #[allow(clippy::too_many_arguments)]
    pub fn append(
        &mut self,
        event_type: EventType,
        subject: Option<String>,
        predicate: Option<String>,
        object: Option<String>,
        graph: Option<String>,
        metadata: HashMap<String, String>,
    ) -> u64 {
        let id = self.counter;
        self.counter += 1;
        self.events.push(RdfStreamEvent {
            id,
            event_type,
            subject,
            predicate,
            object,
            graph,
            timestamp_ms: now_ms(),
            metadata,
        });
        id
    }

    /// Retrieve a single event by id.
    pub fn get(&self, id: u64) -> Option<&RdfStreamEvent> {
        self.events.iter().find(|e| e.id == id)
    }

    /// Return all events whose id falls within `[start_id, end_id]` (inclusive).
    pub fn range(&self, start_id: u64, end_id: u64) -> Vec<&RdfStreamEvent> {
        self.events
            .iter()
            .filter(|e| e.id >= start_id && e.id <= end_id)
            .collect()
    }

    /// Return all events of a given type.
    pub fn by_type(&self, event_type: &EventType) -> Vec<&RdfStreamEvent> {
        self.events
            .iter()
            .filter(|e| &e.event_type == event_type)
            .collect()
    }

    /// Return all events associated with a specific named graph.
    pub fn by_graph(&self, graph: &str) -> Vec<&RdfStreamEvent> {
        self.events
            .iter()
            .filter(|e| e.graph.as_deref() == Some(graph))
            .collect()
    }

    /// Iterate over all events with id >= `from_id`.
    pub fn replay(&self, from_id: u64) -> impl Iterator<Item = &RdfStreamEvent> {
        self.events.iter().filter(move |e| e.id >= from_id)
    }

    /// Compute a snapshot: event counts per type plus first/last ids.
    pub fn snapshot(&self) -> RdfEventSnapshot {
        let mut counts: HashMap<String, u64> = HashMap::new();
        for ev in &self.events {
            *counts.entry(ev.event_type.to_string()).or_insert(0) += 1;
        }
        let total = self.events.len() as u64;
        let first_id = self.events.first().map(|e| e.id);
        let last_id = self.events.last().map(|e| e.id);
        RdfEventSnapshot {
            counts,
            total,
            first_id,
            last_id,
        }
    }

    /// Remove all events with id < `id`; returns the number of events removed.
    pub fn truncate_before(&mut self, id: u64) -> usize {
        let before = self.events.len();
        self.events.retain(|e| e.id >= id);
        before - self.events.len()
    }

    /// Number of events currently held.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// `true` if the store contains no events.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }
}

/// A point-in-time snapshot of event store statistics.
#[derive(Clone, Debug)]
pub struct RdfEventSnapshot {
    /// Count of events per event-type label.
    pub counts: HashMap<String, u64>,
    /// Total number of events.
    pub total: u64,
    /// Id of the oldest retained event, if any.
    pub first_id: Option<u64>,
    /// Id of the most recent event, if any.
    pub last_id: Option<u64>,
}

/// Replays a sequence of events to rebuild derived state.
#[derive(Debug, Default)]
pub struct EventProjector;

impl EventProjector {
    /// Create a new projector.
    pub fn new() -> Self {
        Self
    }

    /// Compute the current set of triples by replaying add/remove events.
    ///
    /// Only events where subject, predicate, and object are all `Some` are
    /// considered. `TripleAdded` inserts; `TripleRemoved` removes.
    pub fn project_triples(events: &[RdfStreamEvent]) -> HashSet<(String, String, String)> {
        let mut set = HashSet::new();
        for ev in events {
            match ev.event_type {
                EventType::TripleAdded => {
                    if let (Some(s), Some(p), Some(o)) = (&ev.subject, &ev.predicate, &ev.object) {
                        set.insert((s.clone(), p.clone(), o.clone()));
                    }
                }
                EventType::TripleRemoved => {
                    if let (Some(s), Some(p), Some(o)) = (&ev.subject, &ev.predicate, &ev.object) {
                        set.remove(&(s.clone(), p.clone(), o.clone()));
                    }
                }
                _ => {}
            }
        }
        set
    }

    /// Compute the current set of named graphs by replaying create/drop events.
    ///
    /// `GraphCreated` inserts the `graph` value; `GraphDropped` removes it.
    pub fn project_graphs(events: &[RdfStreamEvent]) -> HashSet<String> {
        let mut set = HashSet::new();
        for ev in events {
            match ev.event_type {
                EventType::GraphCreated => {
                    if let Some(g) = &ev.graph {
                        set.insert(g.clone());
                    }
                }
                EventType::GraphDropped => {
                    if let Some(g) = &ev.graph {
                        set.remove(g);
                    }
                }
                _ => {}
            }
        }
        set
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn no_meta() -> HashMap<String, String> {
        HashMap::new()
    }

    fn make_store() -> RdfEventStore {
        let mut store = RdfEventStore::new();
        store.append(
            EventType::TripleAdded,
            Some("s1".into()),
            Some("p1".into()),
            Some("o1".into()),
            None,
            no_meta(),
        );
        store.append(
            EventType::TripleAdded,
            Some("s2".into()),
            Some("p2".into()),
            Some("o2".into()),
            Some("g1".into()),
            no_meta(),
        );
        store.append(
            EventType::GraphCreated,
            None,
            None,
            None,
            Some("g1".into()),
            no_meta(),
        );
        store
    }

    #[test]
    fn test_new_store_is_empty() {
        let store = RdfEventStore::new();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn test_append_increments_counter() {
        let mut store = RdfEventStore::new();
        let id0 = store.append(EventType::TripleAdded, None, None, None, None, no_meta());
        let id1 = store.append(EventType::TripleAdded, None, None, None, None, no_meta());
        assert_eq!(id0, 0);
        assert_eq!(id1, 1);
    }

    #[test]
    fn test_len_after_append() {
        let store = make_store();
        assert_eq!(store.len(), 3);
    }

    #[test]
    fn test_is_empty_false_after_append() {
        let store = make_store();
        assert!(!store.is_empty());
    }

    #[test]
    fn test_get_existing_event() {
        let store = make_store();
        let ev = store.get(0).expect("event 0 should exist");
        assert_eq!(ev.event_type, EventType::TripleAdded);
    }

    #[test]
    fn test_get_missing_event_returns_none() {
        let store = make_store();
        assert!(store.get(999).is_none());
    }

    #[test]
    fn test_range_inclusive() {
        let store = make_store();
        let events = store.range(0, 1);
        assert_eq!(events.len(), 2);
    }

    #[test]
    fn test_range_single() {
        let store = make_store();
        let events = store.range(1, 1);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].id, 1);
    }

    #[test]
    fn test_range_empty_when_out_of_bounds() {
        let store = make_store();
        let events = store.range(100, 200);
        assert!(events.is_empty());
    }

    #[test]
    fn test_by_type_triple_added() {
        let store = make_store();
        let added = store.by_type(&EventType::TripleAdded);
        assert_eq!(added.len(), 2);
    }

    #[test]
    fn test_by_type_graph_created() {
        let store = make_store();
        let created = store.by_type(&EventType::GraphCreated);
        assert_eq!(created.len(), 1);
    }

    #[test]
    fn test_by_type_empty_result() {
        let store = make_store();
        let dropped = store.by_type(&EventType::GraphDropped);
        assert!(dropped.is_empty());
    }

    #[test]
    fn test_by_graph_filters_correctly() {
        let store = make_store();
        let g1_events = store.by_graph("g1");
        assert_eq!(g1_events.len(), 2); // TripleAdded(g1) + GraphCreated(g1)
    }

    #[test]
    fn test_by_graph_no_match() {
        let store = make_store();
        assert!(store.by_graph("nonexistent").is_empty());
    }

    #[test]
    fn test_replay_from_zero() {
        let store = make_store();
        let replayed: Vec<_> = store.replay(0).collect();
        assert_eq!(replayed.len(), 3);
    }

    #[test]
    fn test_replay_from_middle() {
        let store = make_store();
        let replayed: Vec<_> = store.replay(2).collect();
        assert_eq!(replayed.len(), 1);
        assert_eq!(replayed[0].id, 2);
    }

    #[test]
    fn test_replay_from_beyond_end_empty() {
        let store = make_store();
        let replayed: Vec<_> = store.replay(100).collect();
        assert!(replayed.is_empty());
    }

    #[test]
    fn test_snapshot_total() {
        let store = make_store();
        let snap = store.snapshot();
        assert_eq!(snap.total, 3);
    }

    #[test]
    fn test_snapshot_first_last_id() {
        let store = make_store();
        let snap = store.snapshot();
        assert_eq!(snap.first_id, Some(0));
        assert_eq!(snap.last_id, Some(2));
    }

    #[test]
    fn test_snapshot_counts_by_type() {
        let store = make_store();
        let snap = store.snapshot();
        assert_eq!(snap.counts.get("TripleAdded").copied().unwrap_or(0), 2);
        assert_eq!(snap.counts.get("GraphCreated").copied().unwrap_or(0), 1);
    }

    #[test]
    fn test_snapshot_empty_store() {
        let store = RdfEventStore::new();
        let snap = store.snapshot();
        assert_eq!(snap.total, 0);
        assert!(snap.first_id.is_none());
        assert!(snap.last_id.is_none());
    }

    #[test]
    fn test_truncate_before_removes_events() {
        let mut store = make_store();
        let removed = store.truncate_before(1);
        assert_eq!(removed, 1);
        assert_eq!(store.len(), 2);
    }

    #[test]
    fn test_truncate_before_zero_removes_none() {
        let mut store = make_store();
        let removed = store.truncate_before(0);
        assert_eq!(removed, 0);
        assert_eq!(store.len(), 3);
    }

    #[test]
    fn test_truncate_before_large_removes_all() {
        let mut store = make_store();
        let removed = store.truncate_before(100);
        assert_eq!(removed, 3);
        assert!(store.is_empty());
    }

    #[test]
    fn test_event_type_display() {
        assert_eq!(EventType::TripleAdded.to_string(), "TripleAdded");
        assert_eq!(EventType::TripleRemoved.to_string(), "TripleRemoved");
        assert_eq!(EventType::GraphCreated.to_string(), "GraphCreated");
        assert_eq!(EventType::GraphDropped.to_string(), "GraphDropped");
        assert_eq!(EventType::DatasetMerged.to_string(), "DatasetMerged");
    }

    #[test]
    fn test_event_type_clone_eq_hash() {
        let t = EventType::TripleAdded;
        let t2 = t.clone();
        assert_eq!(t, t2);
        let mut set = std::collections::HashSet::new();
        set.insert(t);
        assert!(set.contains(&EventType::TripleAdded));
    }

    #[test]
    fn test_project_triples_add_remove() {
        let events = vec![
            RdfStreamEvent {
                id: 0,
                event_type: EventType::TripleAdded,
                subject: Some("s".into()),
                predicate: Some("p".into()),
                object: Some("o".into()),
                graph: None,
                timestamp_ms: 0,
                metadata: HashMap::new(),
            },
            RdfStreamEvent {
                id: 1,
                event_type: EventType::TripleAdded,
                subject: Some("s2".into()),
                predicate: Some("p".into()),
                object: Some("o".into()),
                graph: None,
                timestamp_ms: 0,
                metadata: HashMap::new(),
            },
            RdfStreamEvent {
                id: 2,
                event_type: EventType::TripleRemoved,
                subject: Some("s".into()),
                predicate: Some("p".into()),
                object: Some("o".into()),
                graph: None,
                timestamp_ms: 0,
                metadata: HashMap::new(),
            },
        ];
        let triples = EventProjector::project_triples(&events);
        assert_eq!(triples.len(), 1);
        assert!(triples.contains(&("s2".into(), "p".into(), "o".into())));
        assert!(!triples.contains(&("s".into(), "p".into(), "o".into())));
    }

    #[test]
    fn test_project_triples_ignores_incomplete() {
        let events = vec![RdfStreamEvent {
            id: 0,
            event_type: EventType::TripleAdded,
            subject: Some("s".into()),
            predicate: None, // missing predicate
            object: Some("o".into()),
            graph: None,
            timestamp_ms: 0,
            metadata: HashMap::new(),
        }];
        let triples = EventProjector::project_triples(&events);
        assert!(triples.is_empty());
    }

    #[test]
    fn test_project_graphs_create_drop() {
        let events = vec![
            RdfStreamEvent {
                id: 0,
                event_type: EventType::GraphCreated,
                subject: None,
                predicate: None,
                object: None,
                graph: Some("g1".into()),
                timestamp_ms: 0,
                metadata: HashMap::new(),
            },
            RdfStreamEvent {
                id: 1,
                event_type: EventType::GraphCreated,
                subject: None,
                predicate: None,
                object: None,
                graph: Some("g2".into()),
                timestamp_ms: 0,
                metadata: HashMap::new(),
            },
            RdfStreamEvent {
                id: 2,
                event_type: EventType::GraphDropped,
                subject: None,
                predicate: None,
                object: None,
                graph: Some("g1".into()),
                timestamp_ms: 0,
                metadata: HashMap::new(),
            },
        ];
        let graphs = EventProjector::project_graphs(&events);
        assert_eq!(graphs.len(), 1);
        assert!(graphs.contains("g2"));
        assert!(!graphs.contains("g1"));
    }

    #[test]
    fn test_project_graphs_drop_nonexistent_is_noop() {
        let events = vec![RdfStreamEvent {
            id: 0,
            event_type: EventType::GraphDropped,
            subject: None,
            predicate: None,
            object: None,
            graph: Some("g_gone".into()),
            timestamp_ms: 0,
            metadata: HashMap::new(),
        }];
        let graphs = EventProjector::project_graphs(&events);
        assert!(graphs.is_empty());
    }

    #[test]
    fn test_rdf_stream_event_metadata() {
        let mut store = RdfEventStore::new();
        let mut meta = HashMap::new();
        meta.insert("source".to_string(), "kafka".to_string());
        let id = store.append(EventType::DatasetMerged, None, None, None, None, meta);
        let ev = store.get(id).expect("event should exist");
        assert_eq!(ev.metadata.get("source").map(String::as_str), Some("kafka"));
    }

    #[test]
    fn test_projector_new_is_valid() {
        let _p = EventProjector::new();
    }

    #[test]
    fn test_project_triples_empty_events() {
        let triples = EventProjector::project_triples(&[]);
        assert!(triples.is_empty());
    }

    #[test]
    fn test_project_graphs_empty_events() {
        let graphs = EventProjector::project_graphs(&[]);
        assert!(graphs.is_empty());
    }

    #[test]
    fn test_snapshot_after_truncate() {
        let mut store = make_store();
        store.truncate_before(2);
        let snap = store.snapshot();
        assert_eq!(snap.total, 1);
        assert_eq!(snap.first_id, Some(2));
        assert_eq!(snap.last_id, Some(2));
    }

    #[test]
    fn test_range_all_events() {
        let store = make_store();
        let all = store.range(0, u64::MAX);
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn test_append_returns_monotonic_ids() {
        let mut store = RdfEventStore::new();
        let ids: Vec<u64> = (0..10)
            .map(|_| store.append(EventType::TripleAdded, None, None, None, None, no_meta()))
            .collect();
        for (i, &id) in ids.iter().enumerate() {
            assert_eq!(id, i as u64);
        }
    }
}
