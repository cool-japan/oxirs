//! Manage active GraphQL subscriptions with efficient lifecycle tracking.
//!
//! `SubscriptionManager` is the single authoritative registry of live
//! subscriptions.  It handles:
//!
//! - Creating subscriptions identified by a UUID and backed by a
//!   `tokio::sync::mpsc` channel pair.
//! - Removing subscriptions when clients disconnect or send `stop`.
//! - Routing `ChangeEvent`s to the subset of subscriptions whose filter
//!   matches the event (see `filter` module).
//! - Exposing lightweight statistics for monitoring.
//!
//! Thread-safety is provided by wrapping mutable state in `tokio::sync::RwLock`
//! so the manager can be `Arc`-shared across async tasks without blocking.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::{mpsc, RwLock};
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::subscription::change_tracker::ChangeEvent;
use crate::subscription::filter::SubscriptionFilter;

/// The channel capacity for per-subscription event delivery queues.
const DEFAULT_CHANNEL_CAPACITY: usize = 256;

/// An active subscription held in the manager.
#[derive(Debug)]
pub struct Subscription {
    /// Unique subscription ID (UUID v4).
    pub id: String,
    /// Filter determining which events this subscription receives.
    pub filter: SubscriptionFilter,
    /// Send-side of the delivery channel.
    pub(crate) sender: mpsc::Sender<Arc<ChangeEvent>>,
    /// Wall-clock time at which the subscription was created.
    pub created_at: Instant,
    /// Number of events delivered so far.
    pub delivered_count: u64,
    /// Number of events that matched the filter but were dropped (full channel).
    pub dropped_count: u64,
}

impl Subscription {
    fn new(
        id: impl Into<String>,
        filter: SubscriptionFilter,
        sender: mpsc::Sender<Arc<ChangeEvent>>,
    ) -> Self {
        Self {
            id: id.into(),
            filter,
            sender,
            created_at: Instant::now(),
            delivered_count: 0,
            dropped_count: 0,
        }
    }

    /// Age of this subscription.
    pub fn age(&self) -> std::time::Duration {
        self.created_at.elapsed()
    }
}

/// A lightweight health snapshot of a single subscription.
#[derive(Debug, Clone)]
pub struct SubscriptionSnapshot {
    /// The subscription ID.
    pub id: String,
    /// Events delivered without being dropped.
    pub delivered_count: u64,
    /// Events that matched but could not be delivered (back-pressure).
    pub dropped_count: u64,
    /// How long the subscription has been alive.
    pub age_secs: f64,
}

/// Statistics across all active subscriptions.
#[derive(Debug, Clone)]
pub struct ManagerStats {
    /// Total number of currently active subscriptions.
    pub active_count: usize,
    /// Cumulative events delivered across all subscriptions.
    pub total_delivered: u64,
    /// Cumulative events dropped across all subscriptions.
    pub total_dropped: u64,
}

/// Manages the lifecycle and routing of GraphQL subscriptions.
///
/// # Usage
///
/// ```rust,no_run
/// # use oxirs_gql::subscription::subscription_manager::SubscriptionManager;
/// # use oxirs_gql::subscription::filter::SubscriptionFilter;
/// # #[tokio::main] async fn main() {
/// let manager = SubscriptionManager::new(256);
/// let (id, mut rx) = manager.subscribe(SubscriptionFilter::all()).await;
///
/// // ... publish change events via ChangeTracker and call manager.notify(event)
///
/// manager.unsubscribe(&id).await;
/// # }
/// ```
pub struct SubscriptionManager {
    /// Per-subscription channel capacity.
    channel_capacity: usize,
    /// Active subscriptions keyed by their ID.
    subscriptions: Arc<RwLock<HashMap<String, Subscription>>>,
}

impl std::fmt::Debug for SubscriptionManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SubscriptionManager")
            .field("channel_capacity", &self.channel_capacity)
            .finish()
    }
}

impl SubscriptionManager {
    /// Create a new manager.
    ///
    /// `channel_capacity` is the number of events each subscriber's channel
    /// can buffer before back-pressure causes event dropping.
    pub fn new(channel_capacity: usize) -> Self {
        Self {
            channel_capacity: channel_capacity.max(1),
            subscriptions: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create a manager with a sensible default channel capacity.
    pub fn with_defaults() -> Self {
        Self::new(DEFAULT_CHANNEL_CAPACITY)
    }

    /// Register a new subscription with the given filter.
    ///
    /// Returns `(subscription_id, receiver)`.  The caller owns the receiver
    /// and must read from it to avoid filling the channel.
    pub async fn subscribe(
        &self,
        filter: SubscriptionFilter,
    ) -> (String, mpsc::Receiver<Arc<ChangeEvent>>) {
        let id = Uuid::new_v4().to_string();
        let (tx, rx) = mpsc::channel(self.channel_capacity);

        let subscription = Subscription::new(id.clone(), filter, tx);

        {
            let mut subs = self.subscriptions.write().await;
            subs.insert(id.clone(), subscription);
        }

        info!(subscription_id = %id, "New subscription registered");
        (id, rx)
    }

    /// Remove the subscription with the given `id`.
    ///
    /// Returns `true` if a subscription was found and removed, `false` if no
    /// such subscription exists.
    pub async fn unsubscribe(&self, id: &str) -> bool {
        let mut subs = self.subscriptions.write().await;
        let removed = subs.remove(id).is_some();

        if removed {
            info!(subscription_id = %id, "Subscription removed");
        } else {
            warn!(subscription_id = %id, "Unsubscribe called for unknown subscription");
        }

        removed
    }

    /// Deliver a `ChangeEvent` to every subscription whose filter matches.
    ///
    /// Events are sent via `try_send`; if a subscriber's channel is full the
    /// event is counted as dropped for that subscriber but delivery continues
    /// to other subscribers.  Closed channels trigger automatic cleanup.
    pub async fn notify(&self, event: Arc<ChangeEvent>) {
        // Collect IDs of subscriptions that need cleanup (closed senders).
        let mut to_remove: Vec<String> = Vec::new();

        {
            let mut subs = self.subscriptions.write().await;

            for (id, sub) in subs.iter_mut() {
                if !sub.filter.matches(&event) {
                    continue;
                }

                match sub.sender.try_send(Arc::clone(&event)) {
                    Ok(()) => {
                        sub.delivered_count += 1;
                        debug!(
                            subscription_id = %id,
                            sequence = event.sequence,
                            "Event delivered"
                        );
                    }
                    Err(mpsc::error::TrySendError::Full(_)) => {
                        sub.dropped_count += 1;
                        warn!(
                            subscription_id = %id,
                            sequence = event.sequence,
                            "Event dropped: subscriber channel full"
                        );
                    }
                    Err(mpsc::error::TrySendError::Closed(_)) => {
                        debug!(subscription_id = %id, "Subscriber channel closed; scheduling cleanup");
                        to_remove.push(id.clone());
                    }
                }
            }

            for id in &to_remove {
                subs.remove(id);
                info!(subscription_id = %id, "Cleaned up closed subscription");
            }
        }
    }

    /// Return the number of currently active subscriptions.
    pub async fn active_count(&self) -> usize {
        self.subscriptions.read().await.len()
    }

    /// Return a snapshot of all active subscriptions for monitoring.
    pub async fn snapshots(&self) -> Vec<SubscriptionSnapshot> {
        let subs = self.subscriptions.read().await;
        subs.values()
            .map(|s| SubscriptionSnapshot {
                id: s.id.clone(),
                delivered_count: s.delivered_count,
                dropped_count: s.dropped_count,
                age_secs: s.age().as_secs_f64(),
            })
            .collect()
    }

    /// Return aggregate statistics.
    pub async fn stats(&self) -> ManagerStats {
        let subs = self.subscriptions.read().await;
        let active_count = subs.len();
        let total_delivered: u64 = subs.values().map(|s| s.delivered_count).sum();
        let total_dropped: u64 = subs.values().map(|s| s.dropped_count).sum();
        ManagerStats {
            active_count,
            total_delivered,
            total_dropped,
        }
    }

    /// Remove all active subscriptions.
    pub async fn clear(&self) {
        let mut subs = self.subscriptions.write().await;
        let count = subs.len();
        subs.clear();
        info!(cleared = count, "All subscriptions cleared");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::subscription::change_tracker::{ChangeEvent, ChangeType};
    use crate::subscription::filter::SubscriptionFilter;
    use std::time::Duration;
    use tokio::time::timeout;

    fn insert_event(seq: u64, subject: &str) -> Arc<ChangeEvent> {
        Arc::new(ChangeEvent::new(
            seq,
            ChangeType::Insert,
            subject,
            "http://ex.org/p",
            "http://ex.org/o",
            None,
        ))
    }

    fn manager() -> SubscriptionManager {
        SubscriptionManager::new(64)
    }

    #[tokio::test]
    async fn test_subscribe_returns_unique_ids() {
        let m = manager();
        let (id1, _rx1) = m.subscribe(SubscriptionFilter::all()).await;
        let (id2, _rx2) = m.subscribe(SubscriptionFilter::all()).await;
        assert_ne!(id1, id2);
    }

    #[tokio::test]
    async fn test_active_count_after_subscribe() {
        let m = manager();
        assert_eq!(m.active_count().await, 0);
        let (_id, _rx) = m.subscribe(SubscriptionFilter::all()).await;
        assert_eq!(m.active_count().await, 1);
    }

    #[tokio::test]
    async fn test_unsubscribe_returns_true_for_existing_id() {
        let m = manager();
        let (id, _rx) = m.subscribe(SubscriptionFilter::all()).await;
        assert!(m.unsubscribe(&id).await);
        assert_eq!(m.active_count().await, 0);
    }

    #[tokio::test]
    async fn test_unsubscribe_returns_false_for_unknown_id() {
        let m = manager();
        assert!(!m.unsubscribe("nonexistent-id").await);
    }

    #[tokio::test]
    async fn test_notify_delivers_to_matching_subscriber() {
        let m = manager();
        let (_id, mut rx) = m.subscribe(SubscriptionFilter::all()).await;

        let event = insert_event(1, "http://ex.org/subject");
        m.notify(event).await;

        let received = timeout(Duration::from_millis(100), rx.recv())
            .await
            .expect("no timeout")
            .expect("received");

        assert_eq!(received.sequence, 1);
    }

    #[tokio::test]
    async fn test_notify_does_not_deliver_to_non_matching_subscriber() {
        let m = manager();
        let filter = SubscriptionFilter::builder()
            .subject("http://ex.org/specific")
            .build();
        let (_id, mut rx) = m.subscribe(filter).await;

        let event = insert_event(1, "http://ex.org/other");
        m.notify(event).await;

        let result = timeout(Duration::from_millis(50), rx.recv()).await;
        assert!(
            result.is_err(),
            "Should not have received a non-matching event"
        );
    }

    #[tokio::test]
    async fn test_notify_fanout_to_multiple_subscribers() {
        let m = manager();
        let (_id1, mut rx1) = m.subscribe(SubscriptionFilter::all()).await;
        let (_id2, mut rx2) = m.subscribe(SubscriptionFilter::all()).await;

        let event = insert_event(1, "s");
        m.notify(event).await;

        let r1 = timeout(Duration::from_millis(100), rx1.recv()).await;
        let r2 = timeout(Duration::from_millis(100), rx2.recv()).await;
        assert!(r1.is_ok() && r1.expect("should succeed").is_some());
        assert!(r2.is_ok() && r2.expect("should succeed").is_some());
    }

    #[tokio::test]
    async fn test_stats_delivered_increments() {
        let m = manager();
        let (_id, _rx) = m.subscribe(SubscriptionFilter::all()).await;

        m.notify(insert_event(1, "s")).await;
        m.notify(insert_event(2, "s")).await;

        let stats = m.stats().await;
        assert_eq!(stats.active_count, 1);
        assert_eq!(stats.total_delivered, 2);
        assert_eq!(stats.total_dropped, 0);
    }

    #[tokio::test]
    async fn test_clear_removes_all_subscriptions() {
        let m = manager();
        let _ = m.subscribe(SubscriptionFilter::all()).await;
        let _ = m.subscribe(SubscriptionFilter::all()).await;
        assert_eq!(m.active_count().await, 2);

        m.clear().await;
        assert_eq!(m.active_count().await, 0);
    }

    #[tokio::test]
    async fn test_snapshots_reflects_active_subscriptions() {
        let m = manager();
        let _ = m.subscribe(SubscriptionFilter::all()).await;
        let snaps = m.snapshots().await;
        assert_eq!(snaps.len(), 1);
    }

    #[tokio::test]
    async fn test_closed_channel_subscription_is_cleaned_up() {
        let m = manager();
        let (id, rx) = m.subscribe(SubscriptionFilter::all()).await;
        // Drop the receiver to close the channel.
        drop(rx);

        // Notify should trigger cleanup of the closed subscription.
        m.notify(insert_event(1, "s")).await;

        assert!(
            !m.unsubscribe(&id).await,
            "Should already have been removed"
        );
    }

    #[tokio::test]
    async fn test_filter_by_event_type_insert_only() {
        let m = manager();
        let filter = SubscriptionFilter::inserts_only();
        let (_id, mut rx) = m.subscribe(filter).await;

        let delete_ev = Arc::new(ChangeEvent::new(1, ChangeType::Delete, "s", "p", "o", None));
        m.notify(delete_ev).await;

        let insert_ev = insert_event(2, "s");
        m.notify(insert_ev).await;

        // Only the insert event should arrive.
        let received = timeout(Duration::from_millis(100), rx.recv())
            .await
            .expect("no timeout")
            .expect("received");

        assert_eq!(received.event_type, ChangeType::Insert);
    }
}
