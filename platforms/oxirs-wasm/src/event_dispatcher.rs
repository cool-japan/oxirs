//! # Event Dispatcher
//!
//! A publish/subscribe event dispatching system for WASM browser events.
//!
//! Because WASM unit tests cannot call real JavaScript callbacks, this module
//! simulates dispatch by tracking handler metadata (name, call count, last
//! called timestamp) rather than invoking real function pointers.
//!
//! ## Example
//!
//! ```rust
//! use oxirs_wasm::event_dispatcher::{EventDispatcher, EventType, WasmEvent};
//! use std::collections::HashMap;
//!
//! let mut dispatcher = EventDispatcher::new();
//!
//! let id = dispatcher.subscribe(EventType::QueryCompleted, "on_query_done");
//! assert_eq!(dispatcher.subscription_count(), 1);
//!
//! let event = WasmEvent {
//!     event_type: EventType::QueryCompleted,
//!     payload: HashMap::new(),
//!     timestamp_ms: 1_000,
//!     source: "sparql_engine".to_string(),
//! };
//!
//! let called = dispatcher.dispatch(event);
//! assert_eq!(called, 1);
//! ```

use std::collections::HashMap;

// ─── EventType ────────────────────────────────────────────────────────────────

/// The type of a WASM / browser event.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EventType {
    /// A SPARQL query has started executing.
    QueryStarted,
    /// A SPARQL query has completed successfully.
    QueryCompleted,
    /// A SPARQL query failed.
    QueryError,
    /// RDF data has been loaded into the store.
    DataLoaded,
    /// Existing RDF data has been updated.
    DataUpdated,
    /// A connection to a remote endpoint has been established.
    ConnectionEstablished,
    /// A connection to a remote endpoint was lost.
    ConnectionLost,
    /// An application-defined event with a custom name.
    Custom(String),
}

impl EventType {
    /// Returns a human-readable label for this event type.
    pub fn label(&self) -> String {
        match self {
            EventType::QueryStarted => "QueryStarted".to_string(),
            EventType::QueryCompleted => "QueryCompleted".to_string(),
            EventType::QueryError => "QueryError".to_string(),
            EventType::DataLoaded => "DataLoaded".to_string(),
            EventType::DataUpdated => "DataUpdated".to_string(),
            EventType::ConnectionEstablished => "ConnectionEstablished".to_string(),
            EventType::ConnectionLost => "ConnectionLost".to_string(),
            EventType::Custom(name) => format!("Custom({name})"),
        }
    }
}

// ─── WasmEvent ────────────────────────────────────────────────────────────────

/// A single event dispatched through the [`EventDispatcher`].
#[derive(Debug, Clone)]
pub struct WasmEvent {
    /// The type/topic of this event.
    pub event_type: EventType,
    /// Arbitrary string key-value payload.
    pub payload: HashMap<String, String>,
    /// Unix timestamp in milliseconds when the event was created.
    pub timestamp_ms: u64,
    /// Identifier of the component that emitted this event.
    pub source: String,
}

// ─── SubscriptionId ───────────────────────────────────────────────────────────

/// Opaque handle returned by [`EventDispatcher::subscribe`].
pub type SubscriptionId = u64;

// ─── HandlerStats ─────────────────────────────────────────────────────────────

/// Runtime statistics for a registered handler.
#[derive(Debug, Clone, Default)]
pub struct HandlerStats {
    /// Total number of times this handler has been invoked.
    pub calls: u64,
    /// Timestamp (ms) of the most recent invocation, or `None` if never called.
    pub last_called_ms: Option<u64>,
}

// ─── Subscription entry ───────────────────────────────────────────────────────

/// Internal record for one subscription.
#[derive(Debug, Clone)]
struct Subscription {
    /// The event type this subscription is listening to.
    event_type: EventType,
    /// Logical handler name (for debugging / stats labelling).
    handler_name: String,
    /// Live statistics for this subscription.
    stats: HandlerStats,
}

// ─── EventDispatcher ─────────────────────────────────────────────────────────

/// Pub/sub event dispatcher for WASM browser events.
///
/// Handlers are represented by their name and call-count statistics.
/// `dispatch` simulates invocation by incrementing the call counter and
/// updating `last_called_ms`.
pub struct EventDispatcher {
    /// All active subscriptions keyed by their ID.
    subscriptions: HashMap<SubscriptionId, Subscription>,
    /// Monotonically increasing counter for subscription IDs.
    next_id: SubscriptionId,
}

impl EventDispatcher {
    /// Creates a new, empty `EventDispatcher`.
    pub fn new() -> Self {
        Self {
            subscriptions: HashMap::new(),
            next_id: 1,
        }
    }

    // ── Subscription management ───────────────────────────────────────────────

    /// Subscribes `handler_name` to events of `event_type`.
    ///
    /// Returns a [`SubscriptionId`] that can be used to unsubscribe or query
    /// handler statistics.
    pub fn subscribe(&mut self, event_type: EventType, handler_name: &str) -> SubscriptionId {
        let id = self.next_id;
        self.next_id += 1;
        self.subscriptions.insert(
            id,
            Subscription {
                event_type,
                handler_name: handler_name.to_string(),
                stats: HandlerStats::default(),
            },
        );
        id
    }

    /// Removes the subscription with the given `id`.
    ///
    /// Returns `true` when the subscription was found and removed, `false` when
    /// the `id` was not registered.
    pub fn unsubscribe(&mut self, id: SubscriptionId) -> bool {
        self.subscriptions.remove(&id).is_some()
    }

    // ── Dispatch ──────────────────────────────────────────────────────────────

    /// Dispatches `event` to all handlers subscribed to `event.event_type`.
    ///
    /// Each matching handler's call counter is incremented and
    /// `last_called_ms` is updated to `event.timestamp_ms`.
    ///
    /// Returns the number of handlers that were invoked.
    pub fn dispatch(&mut self, event: WasmEvent) -> usize {
        let mut count = 0usize;
        for sub in self.subscriptions.values_mut() {
            if sub.event_type == event.event_type {
                sub.stats.calls += 1;
                sub.stats.last_called_ms = Some(event.timestamp_ms);
                count += 1;
            }
        }
        count
    }

    /// Dispatches `event` to **all** registered handlers regardless of the
    /// event type they subscribed to.
    ///
    /// Returns the total number of handlers invoked.
    pub fn dispatch_to_all(&mut self, event: WasmEvent) -> usize {
        let count = self.subscriptions.len();
        for sub in self.subscriptions.values_mut() {
            sub.stats.calls += 1;
            sub.stats.last_called_ms = Some(event.timestamp_ms);
        }
        count
    }

    // ── Queries ───────────────────────────────────────────────────────────────

    /// Returns the total number of active subscriptions.
    pub fn subscription_count(&self) -> usize {
        self.subscriptions.len()
    }

    /// Returns the live [`HandlerStats`] for subscription `id`, or `None` when
    /// no such subscription exists.
    pub fn handler_stats(&self, id: SubscriptionId) -> Option<&HandlerStats> {
        self.subscriptions.get(&id).map(|s| &s.stats)
    }

    /// Returns the number of subscriptions listening for `event_type`.
    pub fn subscriptions_for_type(&self, event_type: &EventType) -> usize {
        self.subscriptions
            .values()
            .filter(|s| &s.event_type == event_type)
            .count()
    }

    /// Returns the handler name for subscription `id`, or `None`.
    pub fn handler_name(&self, id: SubscriptionId) -> Option<&str> {
        self.subscriptions.get(&id).map(|s| s.handler_name.as_str())
    }

    /// Removes all subscriptions.
    pub fn clear_all(&mut self) {
        self.subscriptions.clear();
    }
}

impl Default for EventDispatcher {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_event(event_type: EventType, ts: u64) -> WasmEvent {
        WasmEvent {
            event_type,
            payload: HashMap::new(),
            timestamp_ms: ts,
            source: "test".to_string(),
        }
    }

    fn make_event_with_payload(event_type: EventType, ts: u64, key: &str, val: &str) -> WasmEvent {
        let mut payload = HashMap::new();
        payload.insert(key.to_string(), val.to_string());
        WasmEvent {
            event_type,
            payload,
            timestamp_ms: ts,
            source: "test".to_string(),
        }
    }

    // ── subscribe ─────────────────────────────────────────────────────────────

    #[test]
    fn test_subscribe_returns_unique_ids() {
        let mut d = EventDispatcher::new();
        let id1 = d.subscribe(EventType::QueryStarted, "h1");
        let id2 = d.subscribe(EventType::QueryStarted, "h2");
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_subscription_count_increments() {
        let mut d = EventDispatcher::new();
        assert_eq!(d.subscription_count(), 0);
        d.subscribe(EventType::DataLoaded, "h1");
        assert_eq!(d.subscription_count(), 1);
        d.subscribe(EventType::DataUpdated, "h2");
        assert_eq!(d.subscription_count(), 2);
    }

    #[test]
    fn test_subscribe_stores_handler_name() {
        let mut d = EventDispatcher::new();
        let id = d.subscribe(EventType::QueryCompleted, "my_handler");
        assert_eq!(d.handler_name(id), Some("my_handler"));
    }

    // ── unsubscribe ───────────────────────────────────────────────────────────

    #[test]
    fn test_unsubscribe_known_id_returns_true() {
        let mut d = EventDispatcher::new();
        let id = d.subscribe(EventType::QueryError, "h");
        assert!(d.unsubscribe(id));
    }

    #[test]
    fn test_unsubscribe_unknown_id_returns_false() {
        let mut d = EventDispatcher::new();
        assert!(!d.unsubscribe(9999));
    }

    #[test]
    fn test_unsubscribe_decrements_count() {
        let mut d = EventDispatcher::new();
        let id = d.subscribe(EventType::ConnectionEstablished, "h");
        assert_eq!(d.subscription_count(), 1);
        d.unsubscribe(id);
        assert_eq!(d.subscription_count(), 0);
    }

    #[test]
    fn test_unsubscribe_removes_handler_from_type_count() {
        let mut d = EventDispatcher::new();
        let id = d.subscribe(EventType::QueryCompleted, "h");
        assert_eq!(d.subscriptions_for_type(&EventType::QueryCompleted), 1);
        d.unsubscribe(id);
        assert_eq!(d.subscriptions_for_type(&EventType::QueryCompleted), 0);
    }

    #[test]
    fn test_unsubscribe_then_no_dispatch() {
        let mut d = EventDispatcher::new();
        let id = d.subscribe(EventType::DataLoaded, "h");
        d.unsubscribe(id);
        let called = d.dispatch(make_event(EventType::DataLoaded, 100));
        assert_eq!(called, 0);
    }

    // ── dispatch ──────────────────────────────────────────────────────────────

    #[test]
    fn test_dispatch_calls_matching_handler() {
        let mut d = EventDispatcher::new();
        let id = d.subscribe(EventType::QueryCompleted, "on_done");
        let called = d.dispatch(make_event(EventType::QueryCompleted, 500));
        assert_eq!(called, 1);
        let stats = d.handler_stats(id).expect("stats");
        assert_eq!(stats.calls, 1);
    }

    #[test]
    fn test_dispatch_does_not_call_mismatched_handler() {
        let mut d = EventDispatcher::new();
        let id = d.subscribe(EventType::QueryStarted, "on_start");
        let called = d.dispatch(make_event(EventType::QueryCompleted, 100));
        assert_eq!(called, 0);
        let stats = d.handler_stats(id).expect("stats");
        assert_eq!(stats.calls, 0);
    }

    #[test]
    fn test_dispatch_returns_zero_with_no_subscribers() {
        let mut d = EventDispatcher::new();
        let called = d.dispatch(make_event(EventType::DataLoaded, 100));
        assert_eq!(called, 0);
    }

    #[test]
    fn test_dispatch_multiple_handlers_same_type() {
        let mut d = EventDispatcher::new();
        let id1 = d.subscribe(EventType::DataUpdated, "h1");
        let id2 = d.subscribe(EventType::DataUpdated, "h2");
        let id3 = d.subscribe(EventType::DataUpdated, "h3");
        let called = d.dispatch(make_event(EventType::DataUpdated, 1000));
        assert_eq!(called, 3);
        for id in [id1, id2, id3] {
            assert_eq!(d.handler_stats(id).expect("stats").calls, 1);
        }
    }

    #[test]
    fn test_dispatch_updates_last_called_ms() {
        let mut d = EventDispatcher::new();
        let id = d.subscribe(EventType::QueryStarted, "h");
        d.dispatch(make_event(EventType::QueryStarted, 7777));
        let stats = d.handler_stats(id).expect("stats");
        assert_eq!(stats.last_called_ms, Some(7777));
    }

    #[test]
    fn test_dispatch_accumulates_call_count() {
        let mut d = EventDispatcher::new();
        let id = d.subscribe(EventType::QueryCompleted, "h");
        for ts in [100, 200, 300] {
            d.dispatch(make_event(EventType::QueryCompleted, ts));
        }
        assert_eq!(d.handler_stats(id).expect("stats").calls, 3);
    }

    #[test]
    fn test_dispatch_last_called_updated_on_each_call() {
        let mut d = EventDispatcher::new();
        let id = d.subscribe(EventType::QueryCompleted, "h");
        d.dispatch(make_event(EventType::QueryCompleted, 100));
        d.dispatch(make_event(EventType::QueryCompleted, 200));
        assert_eq!(
            d.handler_stats(id).expect("stats").last_called_ms,
            Some(200)
        );
    }

    #[test]
    fn test_dispatch_mixed_types_only_calls_matching() {
        let mut d = EventDispatcher::new();
        let id_a = d.subscribe(EventType::DataLoaded, "ha");
        let id_b = d.subscribe(EventType::DataUpdated, "hb");
        d.dispatch(make_event(EventType::DataLoaded, 50));
        assert_eq!(d.handler_stats(id_a).expect("a").calls, 1);
        assert_eq!(d.handler_stats(id_b).expect("b").calls, 0);
    }

    // ── dispatch_to_all ───────────────────────────────────────────────────────

    #[test]
    fn test_dispatch_to_all_calls_every_handler() {
        let mut d = EventDispatcher::new();
        let id1 = d.subscribe(EventType::QueryStarted, "h1");
        let id2 = d.subscribe(EventType::DataLoaded, "h2");
        let id3 = d.subscribe(EventType::ConnectionLost, "h3");
        let called = d.dispatch_to_all(make_event(EventType::QueryCompleted, 999));
        assert_eq!(called, 3);
        for id in [id1, id2, id3] {
            assert_eq!(d.handler_stats(id).expect("stats").calls, 1);
        }
    }

    #[test]
    fn test_dispatch_to_all_updates_timestamp_for_all() {
        let mut d = EventDispatcher::new();
        let id1 = d.subscribe(EventType::QueryStarted, "h1");
        let id2 = d.subscribe(EventType::DataUpdated, "h2");
        d.dispatch_to_all(make_event(EventType::QueryError, 12345));
        assert_eq!(
            d.handler_stats(id1).expect("s1").last_called_ms,
            Some(12345)
        );
        assert_eq!(
            d.handler_stats(id2).expect("s2").last_called_ms,
            Some(12345)
        );
    }

    #[test]
    fn test_dispatch_to_all_returns_zero_when_empty() {
        let mut d = EventDispatcher::new();
        let called = d.dispatch_to_all(make_event(EventType::DataLoaded, 1));
        assert_eq!(called, 0);
    }

    // ── stats tracking ────────────────────────────────────────────────────────

    #[test]
    fn test_initial_stats_are_zero() {
        let mut d = EventDispatcher::new();
        let id = d.subscribe(EventType::QueryError, "h");
        let stats = d.handler_stats(id).expect("stats");
        assert_eq!(stats.calls, 0);
        assert!(stats.last_called_ms.is_none());
    }

    #[test]
    fn test_handler_stats_returns_none_for_unknown_id() {
        let d = EventDispatcher::new();
        assert!(d.handler_stats(42).is_none());
    }

    #[test]
    fn test_handler_stats_after_unsubscribe_returns_none() {
        let mut d = EventDispatcher::new();
        let id = d.subscribe(EventType::DataLoaded, "h");
        d.unsubscribe(id);
        assert!(d.handler_stats(id).is_none());
    }

    // ── subscriptions_for_type ────────────────────────────────────────────────

    #[test]
    fn test_subscriptions_for_type_zero_initially() {
        let d = EventDispatcher::new();
        assert_eq!(d.subscriptions_for_type(&EventType::QueryCompleted), 0);
    }

    #[test]
    fn test_subscriptions_for_type_counts_correctly() {
        let mut d = EventDispatcher::new();
        d.subscribe(EventType::DataLoaded, "h1");
        d.subscribe(EventType::DataLoaded, "h2");
        d.subscribe(EventType::QueryCompleted, "h3");
        assert_eq!(d.subscriptions_for_type(&EventType::DataLoaded), 2);
        assert_eq!(d.subscriptions_for_type(&EventType::QueryCompleted), 1);
        assert_eq!(d.subscriptions_for_type(&EventType::QueryStarted), 0);
    }

    // ── clear_all ────────────────────────────────────────────────────────────

    #[test]
    fn test_clear_all_removes_all_subscriptions() {
        let mut d = EventDispatcher::new();
        d.subscribe(EventType::DataLoaded, "h1");
        d.subscribe(EventType::DataUpdated, "h2");
        d.subscribe(EventType::QueryError, "h3");
        assert_eq!(d.subscription_count(), 3);
        d.clear_all();
        assert_eq!(d.subscription_count(), 0);
    }

    #[test]
    fn test_clear_all_then_dispatch_returns_zero() {
        let mut d = EventDispatcher::new();
        d.subscribe(EventType::QueryCompleted, "h");
        d.clear_all();
        let called = d.dispatch(make_event(EventType::QueryCompleted, 1));
        assert_eq!(called, 0);
    }

    #[test]
    fn test_clear_all_then_resubscribe_works() {
        let mut d = EventDispatcher::new();
        d.subscribe(EventType::QueryCompleted, "h");
        d.clear_all();
        let id = d.subscribe(EventType::QueryCompleted, "h2");
        let called = d.dispatch(make_event(EventType::QueryCompleted, 1));
        assert_eq!(called, 1);
        assert_eq!(d.handler_stats(id).expect("s").calls, 1);
    }

    // ── Custom event type ─────────────────────────────────────────────────────

    #[test]
    fn test_custom_event_subscribe_and_dispatch() {
        let mut d = EventDispatcher::new();
        let id = d.subscribe(EventType::Custom("my_event".to_string()), "custom_handler");
        let called = d.dispatch(make_event(EventType::Custom("my_event".to_string()), 1));
        assert_eq!(called, 1);
        assert_eq!(d.handler_stats(id).expect("s").calls, 1);
    }

    #[test]
    fn test_custom_event_different_names_do_not_match() {
        let mut d = EventDispatcher::new();
        let id = d.subscribe(EventType::Custom("event_a".to_string()), "ha");
        let called = d.dispatch(make_event(EventType::Custom("event_b".to_string()), 1));
        assert_eq!(called, 0);
        assert_eq!(d.handler_stats(id).expect("s").calls, 0);
    }

    #[test]
    fn test_custom_event_label() {
        let t = EventType::Custom("special_event".to_string());
        assert_eq!(t.label(), "Custom(special_event)");
    }

    #[test]
    fn test_standard_event_labels() {
        assert_eq!(EventType::QueryStarted.label(), "QueryStarted");
        assert_eq!(EventType::QueryCompleted.label(), "QueryCompleted");
        assert_eq!(EventType::QueryError.label(), "QueryError");
        assert_eq!(EventType::DataLoaded.label(), "DataLoaded");
        assert_eq!(EventType::DataUpdated.label(), "DataUpdated");
        assert_eq!(
            EventType::ConnectionEstablished.label(),
            "ConnectionEstablished"
        );
        assert_eq!(EventType::ConnectionLost.label(), "ConnectionLost");
    }

    // ── payload handling ──────────────────────────────────────────────────────

    #[test]
    fn test_dispatch_with_payload() {
        let mut d = EventDispatcher::new();
        let id = d.subscribe(EventType::DataLoaded, "h");
        let event = make_event_with_payload(EventType::DataLoaded, 42, "graph", "default");
        let called = d.dispatch(event);
        assert_eq!(called, 1);
        assert_eq!(d.handler_stats(id).expect("s").calls, 1);
    }

    // ── source field ──────────────────────────────────────────────────────────

    #[test]
    fn test_event_source_field() {
        let event = WasmEvent {
            event_type: EventType::QueryCompleted,
            payload: HashMap::new(),
            timestamp_ms: 100,
            source: "sparql_engine".to_string(),
        };
        assert_eq!(event.source, "sparql_engine");
    }

    // ── default / debug impls ─────────────────────────────────────────────────

    #[test]
    fn test_dispatcher_default_is_empty() {
        let d = EventDispatcher::default();
        assert_eq!(d.subscription_count(), 0);
    }

    #[test]
    fn test_handler_stats_default() {
        let s = HandlerStats::default();
        assert_eq!(s.calls, 0);
        assert!(s.last_called_ms.is_none());
    }

    // ── multiple dispatch cycles ──────────────────────────────────────────────

    #[test]
    fn test_multiple_dispatch_cycles_accumulate() {
        let mut d = EventDispatcher::new();
        let id = d.subscribe(EventType::ConnectionEstablished, "conn_handler");
        for i in 1u64..=10 {
            d.dispatch(make_event(EventType::ConnectionEstablished, i * 100));
        }
        let stats = d.handler_stats(id).expect("s");
        assert_eq!(stats.calls, 10);
        assert_eq!(stats.last_called_ms, Some(1000));
    }

    #[test]
    fn test_connection_lost_event() {
        let mut d = EventDispatcher::new();
        let id = d.subscribe(EventType::ConnectionLost, "on_disconnect");
        let called = d.dispatch(make_event(EventType::ConnectionLost, 9999));
        assert_eq!(called, 1);
        assert_eq!(d.handler_stats(id).expect("s").calls, 1);
    }

    #[test]
    fn test_dispatch_to_all_accumulates_with_normal_dispatch() {
        let mut d = EventDispatcher::new();
        let id = d.subscribe(EventType::QueryStarted, "h");
        d.dispatch(make_event(EventType::QueryStarted, 1));
        d.dispatch_to_all(make_event(EventType::DataLoaded, 2));
        let stats = d.handler_stats(id).expect("s");
        assert_eq!(stats.calls, 2);
    }
}
