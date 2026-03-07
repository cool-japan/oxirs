//! GraphQL Subscription Optimization (v0.3.0)
//!
//! This module provides efficient real-time update delivery for GraphQL
//! subscriptions over an RDF dataset.  It is structured into four focused
//! sub-modules:
//!
//! - [`change_tracker`] — Atomic RDF mutation recording with broadcast fan-out.
//! - [`filter`] — Compile-time-free pattern matching for change events.
//! - [`subscription_manager`] — Lifecycle registry for active subscriptions.
//! - [`broadcaster`] — Async task that wires tracker output to the manager.
//!
//! ## Pre-existing infrastructure (v0.2.0 / v0.1.0)
//!
//! - [`event_bus`] — Broadcast-channel event bus (named-graph aware).
//! - [`multiplexer`] — Fan-out multiplexer with backpressure and replay tokens.

// v0.3.0 — Subscription Optimization
pub mod broadcaster;
pub mod change_tracker;
pub mod filter;
pub mod subscription_manager;

// Pre-existing infrastructure
pub mod event_bus;
pub mod multiplexer;

// --- v0.3.0 re-exports -------------------------------------------------------

pub use broadcaster::{Broadcaster, BroadcasterBuilder, BroadcasterStats};

pub use change_tracker::{
    BatchChangeTracker, ChangeEvent, ChangeTracker, ChangeTrackerStats, ChangeType,
};

pub use filter::{FilterBuilder, MatchStrategy, StringConstraint, SubscriptionFilter};

pub use subscription_manager::{
    ManagerStats, Subscription, SubscriptionManager, SubscriptionSnapshot,
};

// --- Pre-existing re-exports -------------------------------------------------

pub use event_bus::{
    FilteredSubscription, GraphChangeEvent, SubscriptionEventBus, SubscriptionEventType,
    SubscriptionFilter as EventBusFilter,
};

pub use multiplexer::{
    MultiplexerConfig, ResumeToken, SubscriptionEvent, SubscriptionHealth, SubscriptionMultiplexer,
};
