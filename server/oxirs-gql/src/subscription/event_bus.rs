//! GraphQL Subscription Event Bus
//!
//! Provides a broadcast-channel-based event bus for RDF graph change notifications.
//! Supports per-graph subscriptions, filtered subscriptions, and fan-out delivery
//! to all active WebSocket subscribers.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use tokio::sync::broadcast;
use tracing::{debug, warn};

/// Types of RDF graph changes that can trigger subscriptions.
#[derive(Debug, Clone, PartialEq)]
pub enum GraphChangeEvent {
    /// One or more triples were added to a named graph or the default graph.
    TriplesAdded {
        /// The named graph IRI, or `None` for the default graph.
        graph: Option<String>,
        /// The added triples as (subject, predicate, object) IRI/literal strings.
        triples: Vec<(String, String, String)>,
    },
    /// One or more triples were removed.
    TriplesRemoved {
        /// The named graph IRI, or `None` for the default graph.
        graph: Option<String>,
        /// The removed triples as (subject, predicate, object) IRI/literal strings.
        triples: Vec<(String, String, String)>,
    },
    /// A named graph was created.
    GraphCreated {
        /// The IRI of the newly created graph.
        graph: String,
    },
    /// A named graph was dropped.
    GraphDropped {
        /// The IRI of the dropped graph.
        graph: String,
    },
    /// A transaction was committed (batch notification).
    TransactionCommitted {
        /// Unique identifier for the transaction.
        transaction_id: String,
        /// All graphs that were modified in this transaction.
        affected_graphs: Vec<String>,
        /// Total number of triples added.
        added_count: usize,
        /// Total number of triples removed.
        removed_count: usize,
    },
}

impl GraphChangeEvent {
    /// Returns the graph IRIs this event is associated with.
    pub fn affected_graphs(&self) -> Vec<Option<&str>> {
        match self {
            GraphChangeEvent::TriplesAdded { graph, .. }
            | GraphChangeEvent::TriplesRemoved { graph, .. } => {
                vec![graph.as_deref()]
            }
            GraphChangeEvent::GraphCreated { graph } | GraphChangeEvent::GraphDropped { graph } => {
                vec![Some(graph.as_str())]
            }
            GraphChangeEvent::TransactionCommitted {
                affected_graphs, ..
            } => affected_graphs.iter().map(|g| Some(g.as_str())).collect(),
        }
    }

    /// Returns a human-readable event type label.
    pub fn event_type_label(&self) -> &'static str {
        match self {
            GraphChangeEvent::TriplesAdded { .. } => "TriplesAdded",
            GraphChangeEvent::TriplesRemoved { .. } => "TriplesRemoved",
            GraphChangeEvent::GraphCreated { .. } => "GraphCreated",
            GraphChangeEvent::GraphDropped { .. } => "GraphDropped",
            GraphChangeEvent::TransactionCommitted { .. } => "TransactionCommitted",
        }
    }
}

/// Which event types a subscription filter matches.
#[derive(Debug, Clone, PartialEq)]
pub enum SubscriptionEventType {
    /// Triple insertion events.
    Add,
    /// Triple deletion events.
    Remove,
    /// Graph creation and deletion events.
    GraphLifecycle,
    /// Transaction committed events.
    Transaction,
}

/// Filter controlling which `GraphChangeEvent`s a subscriber receives.
///
/// All specified constraints are AND-ed together. An empty list of
/// `event_types` matches all event types.
#[derive(Debug, Clone)]
pub struct SubscriptionFilter {
    /// If set, only events affecting at least one of these graph IRIs pass.
    pub graphs: Option<Vec<String>>,
    /// If set, only triple events whose predicate is in this list pass.
    pub predicates: Option<Vec<String>>,
    /// If set, only triple events whose subject is in this list pass.
    pub subjects: Option<Vec<String>>,
    /// Event types to include. Empty = include all types.
    pub event_types: Vec<SubscriptionEventType>,
}

impl SubscriptionFilter {
    /// Create a filter that accepts every event without restriction.
    pub fn all() -> Self {
        Self {
            graphs: None,
            predicates: None,
            subjects: None,
            event_types: vec![],
        }
    }

    /// Create a filter restricted to a specific named graph.
    pub fn for_graph(graph: &str) -> Self {
        Self {
            graphs: Some(vec![graph.to_string()]),
            predicates: None,
            subjects: None,
            event_types: vec![],
        }
    }

    /// Create a filter restricted to events involving a specific predicate.
    pub fn for_predicate(predicate: &str) -> Self {
        Self {
            graphs: None,
            predicates: Some(vec![predicate.to_string()]),
            subjects: None,
            event_types: vec![],
        }
    }

    /// Create a filter for add events only.
    pub fn adds_only() -> Self {
        Self {
            graphs: None,
            predicates: None,
            subjects: None,
            event_types: vec![SubscriptionEventType::Add],
        }
    }

    /// Create a filter for remove events only.
    pub fn removes_only() -> Self {
        Self {
            graphs: None,
            predicates: None,
            subjects: None,
            event_types: vec![SubscriptionEventType::Remove],
        }
    }

    /// Test whether a given event passes this filter.
    pub fn matches(&self, event: &GraphChangeEvent) -> bool {
        // Check event type constraint
        if !self.event_types.is_empty() {
            let event_type = match event {
                GraphChangeEvent::TriplesAdded { .. } => SubscriptionEventType::Add,
                GraphChangeEvent::TriplesRemoved { .. } => SubscriptionEventType::Remove,
                GraphChangeEvent::GraphCreated { .. } | GraphChangeEvent::GraphDropped { .. } => {
                    SubscriptionEventType::GraphLifecycle
                }
                GraphChangeEvent::TransactionCommitted { .. } => SubscriptionEventType::Transaction,
            };
            if !self.event_types.contains(&event_type) {
                return false;
            }
        }

        // Check graph constraint
        if let Some(ref allowed_graphs) = self.graphs {
            let event_graphs: Vec<Option<&str>> = event.affected_graphs();
            let any_match = event_graphs.iter().any(|eg| {
                eg.map(|g| allowed_graphs.iter().any(|ag| ag == g))
                    .unwrap_or(false)
            });
            if !any_match {
                return false;
            }
        }

        // Check predicate/subject constraints (only for triple events)
        match event {
            GraphChangeEvent::TriplesAdded { triples, .. }
            | GraphChangeEvent::TriplesRemoved { triples, .. } => {
                if let Some(ref allowed_preds) = self.predicates {
                    let any_pred = triples.iter().any(|(_, p, _)| allowed_preds.contains(p));
                    if !any_pred {
                        return false;
                    }
                }
                if let Some(ref allowed_subjects) = self.subjects {
                    let any_sub = triples.iter().any(|(s, _, _)| allowed_subjects.contains(s));
                    if !any_sub {
                        return false;
                    }
                }
            }
            // Non-triple events always pass predicate/subject constraints
            _ => {}
        }

        true
    }
}

/// A subscription receiver pre-filtered by a `SubscriptionFilter`.
///
/// Wraps a `broadcast::Receiver<GraphChangeEvent>` and skips events that do
/// not pass the filter, so callers only see matching events.
pub struct FilteredSubscription {
    inner: broadcast::Receiver<GraphChangeEvent>,
    filter: SubscriptionFilter,
}

impl FilteredSubscription {
    /// Receive the next event that matches the filter.
    ///
    /// Internally drains lagged/non-matching events until a matching one is
    /// found or the channel is closed.
    pub async fn recv(&mut self) -> Result<GraphChangeEvent, broadcast::error::RecvError> {
        loop {
            match self.inner.recv().await {
                Ok(event) => {
                    if self.filter.matches(&event) {
                        return Ok(event);
                    }
                    // Skip non-matching events and continue waiting.
                    debug!(
                        event_type = event.event_type_label(),
                        "Skipping non-matching event in filtered subscription"
                    );
                }
                Err(broadcast::error::RecvError::Lagged(count)) => {
                    warn!("Filtered subscription lagged by {} events", count);
                    // Propagate lag error so caller can decide how to handle it.
                    return Err(broadcast::error::RecvError::Lagged(count));
                }
                Err(broadcast::error::RecvError::Closed) => {
                    return Err(broadcast::error::RecvError::Closed);
                }
            }
        }
    }

    /// Returns a reference to the underlying filter.
    pub fn filter(&self) -> &SubscriptionFilter {
        &self.filter
    }
}

/// Central event bus for GraphQL subscription fan-out.
///
/// The event bus maintains:
/// - A **global channel** that receives every published event.
/// - **Per-graph channels** that receive only events touching a given graph IRI.
///
/// Callers that want fine-grained filtering should use `subscribe_filtered`,
/// which wraps a global receiver in a `FilteredSubscription`.
pub struct SubscriptionEventBus {
    /// Channel capacity (number of buffered events per receiver).
    capacity: usize,
    /// Per-graph broadcast senders keyed by graph IRI.
    topics: Arc<RwLock<HashMap<String, broadcast::Sender<GraphChangeEvent>>>>,
    /// Global sender — every event is sent here.
    global_sender: broadcast::Sender<GraphChangeEvent>,
    /// Total number of global subscriber handles created (for statistics).
    subscriber_count: Arc<AtomicUsize>,
}

impl SubscriptionEventBus {
    /// Create a new event bus with the given broadcast channel capacity.
    ///
    /// `capacity` is the number of events that can be buffered per receiver
    /// before older events are dropped (lagged).
    pub fn new(capacity: usize) -> Self {
        let (global_sender, _) = broadcast::channel(capacity);
        Self {
            capacity,
            topics: Arc::new(RwLock::new(HashMap::new())),
            global_sender,
            subscriber_count: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Publish an event to all matching subscribers.
    ///
    /// Returns the number of active global subscribers the event was sent to.
    /// Per-graph subscribers are notified additionally.
    pub fn publish(&self, event: GraphChangeEvent) -> usize {
        // Fan-out to per-graph topics
        let affected = event.affected_graphs();
        if let Ok(topics) = self.topics.read() {
            for graph_iri in affected.iter().flatten() {
                let key: &str = graph_iri;
                if let Some(sender) = topics.get(key) {
                    let _ = sender.send(event.clone());
                }
            }
        }

        // Fan-out on the global channel (unwrap_or_default handles the no-receivers case)
        self.global_sender.send(event).unwrap_or_default()
    }

    /// Subscribe to all events, returning a `broadcast::Receiver`.
    pub fn subscribe_all(&self) -> broadcast::Receiver<GraphChangeEvent> {
        self.subscriber_count.fetch_add(1, Ordering::Relaxed);
        self.global_sender.subscribe()
    }

    /// Subscribe to events for a specific named graph IRI.
    ///
    /// Creates a per-graph channel on first use. Events published with a
    /// matching graph IRI are forwarded to this receiver.
    pub fn subscribe_graph(&self, graph: &str) -> broadcast::Receiver<GraphChangeEvent> {
        let mut topics = self
            .topics
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        let sender = topics.entry(graph.to_string()).or_insert_with(|| {
            let (s, _) = broadcast::channel(self.capacity);
            s
        });

        sender.subscribe()
    }

    /// Return the total number of active global subscriber handles.
    pub fn subscriber_count(&self) -> usize {
        self.global_sender.receiver_count()
    }

    /// Return the capacity configured for this event bus.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Create a filtered subscription over the global channel.
    pub fn subscribe_filtered(&self, filter: SubscriptionFilter) -> FilteredSubscription {
        FilteredSubscription {
            inner: self.subscribe_all(),
            filter,
        }
    }
}

// Allow Arc-sharing
impl std::fmt::Debug for SubscriptionEventBus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SubscriptionEventBus")
            .field("capacity", &self.capacity)
            .field("subscriber_count", &self.subscriber_count())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{timeout, Duration};

    fn make_triple_added(graph: Option<&str>) -> GraphChangeEvent {
        GraphChangeEvent::TriplesAdded {
            graph: graph.map(|g| g.to_string()),
            triples: vec![(
                "http://ex.org/s".to_string(),
                "http://ex.org/p".to_string(),
                "http://ex.org/o".to_string(),
            )],
        }
    }

    #[test]
    fn test_filter_all_matches_everything() {
        let filter = SubscriptionFilter::all();
        let event = make_triple_added(Some("http://ex.org/graph1"));
        assert!(filter.matches(&event));

        let lifecycle = GraphChangeEvent::GraphCreated {
            graph: "http://ex.org/g".to_string(),
        };
        assert!(filter.matches(&lifecycle));
    }

    #[test]
    fn test_filter_for_graph_matches_correct_graph() {
        let filter = SubscriptionFilter::for_graph("http://ex.org/graph1");
        let matching = make_triple_added(Some("http://ex.org/graph1"));
        assert!(filter.matches(&matching));

        let non_matching = make_triple_added(Some("http://ex.org/other"));
        assert!(!filter.matches(&non_matching));
    }

    #[test]
    fn test_filter_for_predicate_matches_correct_predicate() {
        let filter = SubscriptionFilter::for_predicate("http://ex.org/p");
        let matching = GraphChangeEvent::TriplesAdded {
            graph: None,
            triples: vec![(
                "http://ex.org/s".to_string(),
                "http://ex.org/p".to_string(),
                "http://ex.org/o".to_string(),
            )],
        };
        assert!(filter.matches(&matching));

        let non_matching = GraphChangeEvent::TriplesAdded {
            graph: None,
            triples: vec![(
                "http://ex.org/s".to_string(),
                "http://ex.org/other_pred".to_string(),
                "http://ex.org/o".to_string(),
            )],
        };
        assert!(!filter.matches(&non_matching));
    }

    #[test]
    fn test_filter_event_type_adds_only() {
        let filter = SubscriptionFilter::adds_only();
        let add = make_triple_added(None);
        assert!(filter.matches(&add));

        let remove = GraphChangeEvent::TriplesRemoved {
            graph: None,
            triples: vec![],
        };
        assert!(!filter.matches(&remove));

        let lifecycle = GraphChangeEvent::GraphCreated {
            graph: "g".to_string(),
        };
        assert!(!filter.matches(&lifecycle));
    }

    #[tokio::test]
    async fn test_event_bus_publish_subscribe_all() {
        let bus = SubscriptionEventBus::new(64);
        let mut rx = bus.subscribe_all();

        let event = make_triple_added(Some("http://ex.org/g"));
        bus.publish(event.clone());

        let received = timeout(Duration::from_millis(100), rx.recv())
            .await
            .expect("Should not time out")
            .expect("Should receive event");

        assert_eq!(received, event);
    }

    #[tokio::test]
    async fn test_event_bus_publish_graph_subscription() {
        let bus = SubscriptionEventBus::new(64);
        let mut graph_rx = bus.subscribe_graph("http://ex.org/graph1");

        // Publish to the correct graph
        bus.publish(GraphChangeEvent::TriplesAdded {
            graph: Some("http://ex.org/graph1".to_string()),
            triples: vec![],
        });

        let received = timeout(Duration::from_millis(100), graph_rx.recv())
            .await
            .expect("Should not time out")
            .expect("Should receive");
        matches!(received, GraphChangeEvent::TriplesAdded { .. });
    }

    #[tokio::test]
    async fn test_event_bus_subscriber_count() {
        let bus = SubscriptionEventBus::new(64);
        assert_eq!(bus.subscriber_count(), 0);

        let _rx1 = bus.subscribe_all();
        let _rx2 = bus.subscribe_all();
        assert_eq!(bus.subscriber_count(), 2);
    }

    #[tokio::test]
    async fn test_filtered_subscription_skips_non_matching() {
        let bus = SubscriptionEventBus::new(64);
        let filter = SubscriptionFilter::for_graph("http://ex.org/target");
        let mut filtered = bus.subscribe_filtered(filter);

        // Publish a non-matching event first, then a matching one.
        bus.publish(make_triple_added(Some("http://ex.org/other")));
        bus.publish(make_triple_added(Some("http://ex.org/target")));

        let received = timeout(Duration::from_millis(200), filtered.recv())
            .await
            .expect("Should not time out")
            .expect("Should receive matching event");

        match received {
            GraphChangeEvent::TriplesAdded { graph, .. } => {
                assert_eq!(graph.as_deref(), Some("http://ex.org/target"));
            }
            _ => panic!("Unexpected event type"),
        }
    }

    #[test]
    fn test_graph_change_event_type_label() {
        assert_eq!(make_triple_added(None).event_type_label(), "TriplesAdded");
        assert_eq!(
            GraphChangeEvent::GraphCreated {
                graph: "g".to_string()
            }
            .event_type_label(),
            "GraphCreated"
        );
    }
}
