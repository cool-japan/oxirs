//! Broadcast `ChangeEvent`s from a `ChangeTracker` to a `SubscriptionManager`.
//!
//! `Broadcaster` sits between the data layer and the subscription layer.  It:
//!
//! 1. Subscribes to the raw `broadcast::Receiver<Arc<ChangeEvent>>` produced by
//!    a `ChangeTracker`.
//! 2. Forwards every received event to a `SubscriptionManager::notify` call,
//!    which in turn filters and delivers the event to matching subscriptions.
//! 3. Runs as a self-contained `tokio` task so the caller is not blocked.
//!
//! Multiple `Broadcaster` instances can share the same `SubscriptionManager`
//! via `Arc` – useful for fan-in from multiple datasets or named graphs.
//!
//! # Backpressure
//!
//! The broadcaster's inner `ChangeTracker` channel uses Tokio's `broadcast`
//! semantics: if the broadcaster falls behind (lagged), it skips the missed
//! events and continues.  The `SubscriptionManager`'s per-subscriber `mpsc`
//! channels are used for final delivery and have their own independent
//! backpressure.

use std::sync::Arc;

use tokio::sync::broadcast;
use tokio::task::JoinHandle;
use tracing::{debug, info, warn};

use crate::subscription::change_tracker::ChangeEvent;
use crate::subscription::subscription_manager::SubscriptionManager;

/// Statistics reported by a running `Broadcaster`.
#[derive(Debug, Clone)]
pub struct BroadcasterStats {
    /// Events received from the source channel.
    pub received: u64,
    /// Events forwarded to the subscription manager.
    pub forwarded: u64,
    /// Events skipped because the source channel lagged.
    pub lagged: u64,
}

/// Internal state shared between the `Broadcaster` handle and the background task.
#[derive(Debug)]
struct BroadcasterState {
    received: std::sync::atomic::AtomicU64,
    forwarded: std::sync::atomic::AtomicU64,
    lagged: std::sync::atomic::AtomicU64,
}

impl BroadcasterState {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            received: std::sync::atomic::AtomicU64::new(0),
            forwarded: std::sync::atomic::AtomicU64::new(0),
            lagged: std::sync::atomic::AtomicU64::new(0),
        })
    }

    fn snapshot(&self) -> BroadcasterStats {
        BroadcasterStats {
            received: self.received.load(std::sync::atomic::Ordering::Relaxed),
            forwarded: self.forwarded.load(std::sync::atomic::Ordering::Relaxed),
            lagged: self.lagged.load(std::sync::atomic::Ordering::Relaxed),
        }
    }
}

/// A running broadcaster that forwards `ChangeEvent`s to subscribers.
///
/// Dropping the `Broadcaster` handle does **not** stop the background task –
/// call [`Broadcaster::shutdown`] explicitly or await the `JoinHandle` returned
/// by [`Broadcaster::into_join_handle`].
pub struct Broadcaster {
    state: Arc<BroadcasterState>,
    shutdown_tx: tokio::sync::oneshot::Sender<()>,
    task: JoinHandle<()>,
}

impl std::fmt::Debug for Broadcaster {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Broadcaster")
            .field("stats", &self.state.snapshot())
            .finish()
    }
}

impl Broadcaster {
    /// Spawn a broadcaster that consumes events from `source` and pushes them
    /// to `manager`.
    ///
    /// The background task runs until:
    /// - `shutdown` is called, or
    /// - the source broadcast channel is closed, or
    /// - the task is dropped.
    pub fn spawn(
        source: broadcast::Receiver<Arc<ChangeEvent>>,
        manager: Arc<SubscriptionManager>,
    ) -> Self {
        let state = BroadcasterState::new();
        let state_clone = Arc::clone(&state);
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();

        let task = tokio::spawn(run_broadcaster(source, manager, state_clone, shutdown_rx));

        Self {
            state,
            shutdown_tx,
            task,
        }
    }

    /// Send a shutdown signal to the background task.
    ///
    /// Returns immediately; use `into_join_handle` to await actual termination.
    pub fn shutdown(self) {
        // Ignore the error if the task already exited.
        let _ = self.shutdown_tx.send(());
    }

    /// Consume the broadcaster and return the underlying `JoinHandle`.
    pub fn into_join_handle(self) -> JoinHandle<()> {
        self.task
    }

    /// Return a statistics snapshot.
    pub fn stats(&self) -> BroadcasterStats {
        self.state.snapshot()
    }
}

/// The background task that drives event routing.
async fn run_broadcaster(
    mut source: broadcast::Receiver<Arc<ChangeEvent>>,
    manager: Arc<SubscriptionManager>,
    state: Arc<BroadcasterState>,
    mut shutdown_rx: tokio::sync::oneshot::Receiver<()>,
) {
    info!("Broadcaster task started");

    loop {
        tokio::select! {
            // Listen for shutdown signal.
            _ = &mut shutdown_rx => {
                info!("Broadcaster received shutdown signal");
                break;
            }

            // Receive the next change event from the tracker.
            result = source.recv() => {
                match result {
                    Ok(event) => {
                        state.received.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        debug!(sequence = event.sequence, event_type = %event.event_type, "Broadcaster forwarding event");

                        manager.notify(event).await;
                        state.forwarded.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                    Err(broadcast::error::RecvError::Lagged(count)) => {
                        warn!(count, "Broadcaster lagged; {} events were missed", count);
                        state.lagged.fetch_add(count, std::sync::atomic::Ordering::Relaxed);
                        // Continue: the next recv() will return the oldest available event.
                    }
                    Err(broadcast::error::RecvError::Closed) => {
                        info!("Source broadcast channel closed; broadcaster exiting");
                        break;
                    }
                }
            }
        }
    }

    info!("Broadcaster task stopped");
}

/// A convenience builder that wires a `ChangeTracker` directly to a
/// `SubscriptionManager` without exposing the raw channel.
///
/// ```rust,no_run
/// # use std::sync::Arc;
/// # use oxirs_gql::subscription::change_tracker::ChangeTracker;
/// # use oxirs_gql::subscription::subscription_manager::SubscriptionManager;
/// # use oxirs_gql::subscription::broadcaster::BroadcasterBuilder;
/// # #[tokio::main] async fn main() {
/// let tracker = Arc::new(ChangeTracker::new(512));
/// let manager = Arc::new(SubscriptionManager::with_defaults());
///
/// let broadcaster = BroadcasterBuilder::new()
///     .tracker(Arc::clone(&tracker))
///     .manager(Arc::clone(&manager))
///     .build();
/// # }
/// ```
pub struct BroadcasterBuilder {
    tracker: Option<Arc<crate::subscription::change_tracker::ChangeTracker>>,
    manager: Option<Arc<SubscriptionManager>>,
}

impl BroadcasterBuilder {
    /// Create an empty builder.
    pub fn new() -> Self {
        Self {
            tracker: None,
            manager: None,
        }
    }

    /// Set the source `ChangeTracker`.
    pub fn tracker(
        mut self,
        tracker: Arc<crate::subscription::change_tracker::ChangeTracker>,
    ) -> Self {
        self.tracker = Some(tracker);
        self
    }

    /// Set the target `SubscriptionManager`.
    pub fn manager(mut self, manager: Arc<SubscriptionManager>) -> Self {
        self.manager = Some(manager);
        self
    }

    /// Build and spawn the broadcaster.
    ///
    /// # Panics
    ///
    /// Panics if `tracker` or `manager` was not set.
    pub fn build(self) -> Broadcaster {
        let tracker = self
            .tracker
            .expect("BroadcasterBuilder: tracker must be set before build()");
        let manager = self
            .manager
            .expect("BroadcasterBuilder: manager must be set before build()");

        let receiver = tracker.subscribe();
        Broadcaster::spawn(receiver, manager)
    }
}

impl Default for BroadcasterBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::subscription::change_tracker::ChangeTracker;
    use crate::subscription::filter::SubscriptionFilter;
    use std::time::Duration;
    use tokio::time::{sleep, timeout};

    fn make_stack() -> (Arc<ChangeTracker>, Arc<SubscriptionManager>, Broadcaster) {
        let tracker = Arc::new(ChangeTracker::new(128));
        let manager = Arc::new(SubscriptionManager::with_defaults());
        let broadcaster = BroadcasterBuilder::new()
            .tracker(Arc::clone(&tracker))
            .manager(Arc::clone(&manager))
            .build();
        (tracker, manager, broadcaster)
    }

    #[tokio::test]
    async fn test_broadcaster_forwards_event_end_to_end() {
        let (tracker, manager, _broadcaster) = make_stack();

        let (_id, mut rx) = manager.subscribe(SubscriptionFilter::all()).await;

        // Give the broadcaster task a moment to start up.
        sleep(Duration::from_millis(5)).await;

        tracker.record_insert("http://ex.org/s", "http://ex.org/p", "obj", None);

        let received = timeout(Duration::from_millis(200), rx.recv())
            .await
            .expect("no timeout")
            .expect("received event");

        assert_eq!(received.subject, "http://ex.org/s");
    }

    #[tokio::test]
    async fn test_broadcaster_stats_increment() {
        let (tracker, manager, broadcaster) = make_stack();

        let (_id, _rx) = manager.subscribe(SubscriptionFilter::all()).await;
        sleep(Duration::from_millis(5)).await;

        tracker.record_insert("s", "p", "o", None);
        tracker.record_delete("s", "p", "o", None);

        // Allow the async task to process events.
        sleep(Duration::from_millis(50)).await;

        let stats = broadcaster.stats();
        assert!(stats.received >= 2, "Expected at least 2 received events");
        assert!(stats.forwarded >= 2, "Expected at least 2 forwarded events");
    }

    #[tokio::test]
    async fn test_broadcaster_fanout_to_multiple_subscribers() {
        let (tracker, manager, _broadcaster) = make_stack();

        let (_id1, mut rx1) = manager.subscribe(SubscriptionFilter::all()).await;
        let (_id2, mut rx2) = manager.subscribe(SubscriptionFilter::all()).await;

        sleep(Duration::from_millis(5)).await;

        tracker.record_insert("s", "p", "o", None);

        let r1 = timeout(Duration::from_millis(200), rx1.recv()).await;
        let r2 = timeout(Duration::from_millis(200), rx2.recv()).await;
        assert!(
            r1.is_ok() && r1.expect("should succeed").is_some(),
            "rx1 should receive"
        );
        assert!(
            r2.is_ok() && r2.expect("should succeed").is_some(),
            "rx2 should receive"
        );
    }

    #[tokio::test]
    async fn test_broadcaster_filtered_fanout() {
        let (tracker, manager, _broadcaster) = make_stack();

        // Only subscribe to insert events.
        let insert_filter = SubscriptionFilter::inserts_only();
        let (_id, mut rx) = manager.subscribe(insert_filter).await;

        sleep(Duration::from_millis(5)).await;

        // Publish a delete – should not be received.
        tracker.record_delete("s", "p", "o", None);

        // Publish an insert – should be received.
        tracker.record_insert("s2", "p", "o", None);

        let received = timeout(Duration::from_millis(200), rx.recv())
            .await
            .expect("no timeout")
            .expect("received");

        // The first event received must be the insert (delete was filtered out).
        assert_eq!(
            received.event_type,
            crate::subscription::change_tracker::ChangeType::Insert
        );
    }

    #[tokio::test]
    async fn test_broadcaster_shutdown_stops_task() {
        let tracker = Arc::new(ChangeTracker::new(128));
        let manager = Arc::new(SubscriptionManager::with_defaults());

        let receiver = tracker.subscribe();
        let broadcaster = Broadcaster::spawn(receiver, Arc::clone(&manager));

        sleep(Duration::from_millis(5)).await;

        let handle = broadcaster.into_join_handle();
        // At this point we've consumed the broadcaster, so the shutdown_tx was dropped too.
        // The task should not be finished yet.
        assert!(!handle.is_finished());
    }

    #[tokio::test]
    async fn test_broadcaster_builder_defaults() {
        let tracker = Arc::new(ChangeTracker::new(64));
        let manager = Arc::new(SubscriptionManager::with_defaults());

        let _broadcaster = BroadcasterBuilder::default()
            .tracker(Arc::clone(&tracker))
            .manager(Arc::clone(&manager))
            .build();

        // Verify initial stats are zero.
        let stats = _broadcaster.stats();
        assert_eq!(stats.received, 0);
        assert_eq!(stats.forwarded, 0);
        assert_eq!(stats.lagged, 0);
    }
}
