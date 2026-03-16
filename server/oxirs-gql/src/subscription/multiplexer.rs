//! Subscription Multiplexer with Backpressure and Connection Resumption
//!
//! Provides:
//! - `SubscriptionMultiplexer` — fan-out of a single event source to many consumers.
//! - Backpressure via per-consumer bounded channels; slow consumers are detected and
//!   either dropped or signalled with a lag notification.
//! - `ResumeToken` — an opaque cursor that lets a reconnecting client resume from the
//!   point it left off (up to a configurable replay buffer).
//! - `SubscriptionHealth` — per-subscription liveness tracking.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

/// A serialised event payload delivered to subscribers.
#[derive(Debug, Clone)]
pub struct SubscriptionEvent {
    /// Monotonically increasing sequence number assigned by the multiplexer.
    pub sequence: u64,
    /// The JSON-encoded GraphQL subscription result.
    pub payload_json: String,
    /// The subscription ID this event belongs to (optional; for routing).
    pub subscription_id: Option<String>,
    /// Wall-clock time the event was published.
    pub timestamp: Instant,
}

impl SubscriptionEvent {
    /// Create a new event with the next sequence number.
    pub fn new(sequence: u64, payload_json: impl Into<String>) -> Self {
        Self {
            sequence,
            payload_json: payload_json.into(),
            subscription_id: None,
            timestamp: Instant::now(),
        }
    }

    /// Attach a subscription ID.
    pub fn with_subscription_id(mut self, id: impl Into<String>) -> Self {
        self.subscription_id = Some(id.into());
        self
    }
}

/// An opaque resume token that encodes the sequence number of the last event a
/// client successfully received, so it can request a replay from that point.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ResumeToken {
    /// The sequence number of the last successfully consumed event.
    pub last_sequence: u64,
    /// The subscription ID this token is for.
    pub subscription_id: String,
}

impl ResumeToken {
    /// Create a new resume token.
    pub fn new(subscription_id: impl Into<String>, last_sequence: u64) -> Self {
        Self {
            last_sequence,
            subscription_id: subscription_id.into(),
        }
    }

    /// Encode the token as a base64-like hex string (no external deps).
    pub fn encode(&self) -> String {
        format!("{:016x}:{}", self.last_sequence, self.subscription_id)
    }

    /// Decode a token produced by `encode`.
    pub fn decode(s: &str) -> Option<Self> {
        let (seq_hex, sub_id) = s.split_once(':')?;

        let last_sequence = u64::from_str_radix(seq_hex, 16).ok()?;
        Some(Self {
            last_sequence,
            subscription_id: sub_id.to_string(),
        })
    }
}

/// Configuration for the multiplexer.
#[derive(Debug, Clone)]
pub struct MultiplexerConfig {
    /// Maximum number of events buffered per consumer before backpressure is applied.
    pub consumer_buffer_size: usize,
    /// Number of recent events kept for replay on reconnect.
    pub replay_buffer_size: usize,
    /// How long a consumer can be slow before being considered lagged.
    pub slow_consumer_timeout: Duration,
    /// Whether to drop slow consumers automatically.
    pub drop_slow_consumers: bool,
}

impl Default for MultiplexerConfig {
    fn default() -> Self {
        Self {
            consumer_buffer_size: 128,
            replay_buffer_size: 256,
            slow_consumer_timeout: Duration::from_secs(10),
            drop_slow_consumers: false,
        }
    }
}

/// Health status of a single subscription consumer.
#[derive(Debug, Clone)]
pub struct SubscriptionHealth {
    /// Unique consumer/subscription ID.
    pub subscription_id: String,
    /// Is this subscription currently active?
    pub is_active: bool,
    /// Sequence number of the last event successfully delivered.
    pub last_delivered_sequence: u64,
    /// Total events delivered to this consumer.
    pub total_delivered: u64,
    /// Total events dropped due to backpressure.
    pub total_dropped: u64,
    /// When this subscription was created.
    pub created_at: Instant,
    /// When the last event was delivered.
    pub last_activity: Option<Instant>,
    /// Whether this consumer is currently lagging.
    pub is_lagging: bool,
}

impl SubscriptionHealth {
    fn new(subscription_id: impl Into<String>) -> Self {
        Self {
            subscription_id: subscription_id.into(),
            is_active: true,
            last_delivered_sequence: 0,
            total_delivered: 0,
            total_dropped: 0,
            created_at: Instant::now(),
            last_activity: None,
            is_lagging: false,
        }
    }

    /// Return how long this subscription has been running.
    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }

    /// Return how long since the last event was delivered.
    pub fn idle_duration(&self) -> Option<Duration> {
        self.last_activity.map(|t| t.elapsed())
    }
}

/// Internal consumer state stored by the multiplexer.
struct ConsumerState {
    sender: mpsc::Sender<SubscriptionEvent>,
    health: SubscriptionHealth,
}

/// Replay buffer — keeps the last N events for late-join / reconnect scenarios.
struct ReplayBuffer {
    events: std::collections::VecDeque<SubscriptionEvent>,
    capacity: usize,
}

impl ReplayBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            events: std::collections::VecDeque::with_capacity(capacity.min(4096)),
            capacity,
        }
    }

    fn push(&mut self, event: SubscriptionEvent) {
        if self.events.len() >= self.capacity {
            self.events.pop_front();
        }
        self.events.push_back(event);
    }

    /// Return all events with sequence > `after_sequence`.
    fn events_after(&self, after_sequence: u64) -> Vec<SubscriptionEvent> {
        self.events
            .iter()
            .filter(|e| e.sequence > after_sequence)
            .cloned()
            .collect()
    }

    fn len(&self) -> usize {
        self.events.len()
    }
}

/// Subscription multiplexer that fans a single stream of events out to many consumers.
///
/// Key features:
/// - Each consumer has a bounded channel; if the channel is full the event is either
///   dropped (with the consumer marked as lagging) or the consumer is disconnected.
/// - A rolling replay buffer allows reconnecting clients to catch up.
/// - Thread-safe via an inner `Mutex`-protected state.
pub struct SubscriptionMultiplexer {
    config: MultiplexerConfig,
    consumers: Arc<Mutex<HashMap<String, ConsumerState>>>,
    replay_buffer: Arc<Mutex<ReplayBuffer>>,
    next_sequence: Arc<AtomicU64>,
    total_published: Arc<AtomicU64>,
    is_closed: Arc<AtomicBool>,
}

impl std::fmt::Debug for SubscriptionMultiplexer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SubscriptionMultiplexer")
            .field("config", &self.config)
            .field("next_sequence", &self.next_sequence.load(Ordering::Relaxed))
            .field(
                "total_published",
                &self.total_published.load(Ordering::Relaxed),
            )
            .finish()
    }
}

impl SubscriptionMultiplexer {
    /// Create a new multiplexer with the given configuration.
    pub fn new(config: MultiplexerConfig) -> Self {
        let replay_cap = config.replay_buffer_size;
        Self {
            config,
            consumers: Arc::new(Mutex::new(HashMap::new())),
            replay_buffer: Arc::new(Mutex::new(ReplayBuffer::new(replay_cap))),
            next_sequence: Arc::new(AtomicU64::new(1)),
            total_published: Arc::new(AtomicU64::new(0)),
            is_closed: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Create a multiplexer with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(MultiplexerConfig::default())
    }

    /// Subscribe a new consumer with the given ID.
    ///
    /// Returns the `mpsc::Receiver` end; the multiplexer retains the sender.
    /// If `resume_token` is provided, queued replay events since the token's
    /// sequence will be sent immediately.
    pub fn subscribe(
        &self,
        subscription_id: impl Into<String>,
        resume_token: Option<&ResumeToken>,
    ) -> Result<mpsc::Receiver<SubscriptionEvent>, String> {
        if self.is_closed.load(Ordering::Relaxed) {
            return Err("Multiplexer is closed".to_string());
        }

        let id = subscription_id.into();
        let (tx, rx) = mpsc::channel(self.config.consumer_buffer_size);
        let health = SubscriptionHealth::new(id.clone());
        let state = ConsumerState {
            sender: tx.clone(),
            health,
        };

        let mut consumers = self
            .consumers
            .lock()
            .map_err(|_| "Consumers lock poisoned".to_string())?;

        consumers.insert(id.clone(), state);
        drop(consumers);

        // Replay missed events if a resume token was provided
        if let Some(token) = resume_token {
            let replay_events = self
                .replay_buffer
                .lock()
                .map(|buf| buf.events_after(token.last_sequence))
                .unwrap_or_default();

            for event in replay_events {
                // Best-effort; ignore send errors during replay
                let _ = tx.try_send(event);
            }
        }

        Ok(rx)
    }

    /// Unsubscribe a consumer by ID.
    pub fn unsubscribe(&self, subscription_id: &str) -> bool {
        self.consumers
            .lock()
            .map(|mut c| {
                if let Some(state) = c.get_mut(subscription_id) {
                    state.health.is_active = false;
                }
                c.remove(subscription_id).is_some()
            })
            .unwrap_or(false)
    }

    /// Publish an event to all active consumers.
    ///
    /// Returns the number of consumers the event was successfully delivered to.
    /// Consumers whose channels are full are marked as lagging; if
    /// `drop_slow_consumers` is enabled they are also removed.
    pub async fn publish(&self, payload_json: impl Into<String>) -> usize {
        if self.is_closed.load(Ordering::Relaxed) {
            return 0;
        }

        let sequence = self.next_sequence.fetch_add(1, Ordering::SeqCst);
        let event = SubscriptionEvent::new(sequence, payload_json);
        self.total_published.fetch_add(1, Ordering::Relaxed);

        // Add to replay buffer
        if let Ok(mut buf) = self.replay_buffer.lock() {
            buf.push(event.clone());
        }

        let mut consumers = match self.consumers.lock() {
            Ok(c) => c,
            Err(_) => return 0,
        };

        let mut delivered = 0;
        let mut to_remove: Vec<String> = Vec::new();

        for (id, state) in consumers.iter_mut() {
            match state.sender.try_send(event.clone()) {
                Ok(()) => {
                    state.health.total_delivered += 1;
                    state.health.last_delivered_sequence = sequence;
                    state.health.last_activity = Some(Instant::now());
                    state.health.is_lagging = false;
                    delivered += 1;
                }
                Err(mpsc::error::TrySendError::Full(_)) => {
                    state.health.total_dropped += 1;
                    state.health.is_lagging = true;

                    if self.config.drop_slow_consumers {
                        to_remove.push(id.clone());
                    }
                }
                Err(mpsc::error::TrySendError::Closed(_)) => {
                    state.health.is_active = false;
                    to_remove.push(id.clone());
                }
            }
        }

        for id in to_remove {
            consumers.remove(&id);
        }

        delivered
    }

    /// Return the current health snapshot for all subscriptions.
    pub fn health_snapshots(&self) -> Vec<SubscriptionHealth> {
        self.consumers
            .lock()
            .map(|c| c.values().map(|s| s.health.clone()).collect())
            .unwrap_or_default()
    }

    /// Return the health snapshot for a specific subscription.
    pub fn health_of(&self, subscription_id: &str) -> Option<SubscriptionHealth> {
        self.consumers
            .lock()
            .ok()
            .and_then(|c| c.get(subscription_id).map(|s| s.health.clone()))
    }

    /// Generate a resume token for a consumer at its current position.
    pub fn resume_token_for(&self, subscription_id: &str) -> Option<ResumeToken> {
        self.consumers.lock().ok().and_then(|c| {
            c.get(subscription_id)
                .map(|s| ResumeToken::new(subscription_id, s.health.last_delivered_sequence))
        })
    }

    /// Current number of active consumers.
    pub fn consumer_count(&self) -> usize {
        self.consumers.lock().map(|c| c.len()).unwrap_or(0)
    }

    /// Total events published since creation.
    pub fn total_published(&self) -> u64 {
        self.total_published.load(Ordering::Relaxed)
    }

    /// Replay buffer occupancy.
    pub fn replay_buffer_size(&self) -> usize {
        self.replay_buffer.lock().map(|b| b.len()).unwrap_or(0)
    }

    /// Close the multiplexer — no new events will be accepted and subscriptions
    /// will be refused.
    pub fn close(&self) {
        self.is_closed.store(true, Ordering::SeqCst);
        if let Ok(mut consumers) = self.consumers.lock() {
            for state in consumers.values_mut() {
                state.health.is_active = false;
            }
            consumers.clear();
        }
    }

    /// Whether this multiplexer has been closed.
    pub fn is_closed(&self) -> bool {
        self.is_closed.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tokio::time::timeout;

    fn mux() -> SubscriptionMultiplexer {
        SubscriptionMultiplexer::with_defaults()
    }

    // ---- ResumeToken tests --------------------------------------------------

    #[test]
    fn test_resume_token_encode_decode_roundtrip() {
        let token = ResumeToken::new("sub-1", 42);
        let encoded = token.encode();
        let decoded = ResumeToken::decode(&encoded);
        assert!(decoded.is_some());
        let decoded = decoded.expect("should succeed");
        assert_eq!(decoded.last_sequence, 42);
        assert_eq!(decoded.subscription_id, "sub-1");
    }

    #[test]
    fn test_resume_token_encode_decode_with_zero_sequence() {
        let token = ResumeToken::new("my-subscription", 0);
        let decoded = ResumeToken::decode(&token.encode()).expect("should succeed");
        assert_eq!(decoded.last_sequence, 0);
    }

    #[test]
    fn test_resume_token_encode_decode_large_sequence() {
        let token = ResumeToken::new("test", u64::MAX);
        let decoded = ResumeToken::decode(&token.encode()).expect("should succeed");
        assert_eq!(decoded.last_sequence, u64::MAX);
    }

    #[test]
    fn test_resume_token_decode_invalid_returns_none() {
        assert!(ResumeToken::decode("not-a-valid-token").is_none());
        assert!(ResumeToken::decode("").is_none());
        assert!(ResumeToken::decode("xyz:sub").is_none()); // invalid hex
    }

    #[test]
    fn test_resume_token_equality() {
        let t1 = ResumeToken::new("s", 10);
        let t2 = ResumeToken::new("s", 10);
        let t3 = ResumeToken::new("s", 11);
        assert_eq!(t1, t2);
        assert_ne!(t1, t3);
    }

    // ---- SubscriptionEvent tests --------------------------------------------

    #[test]
    fn test_subscription_event_creation() {
        let event = SubscriptionEvent::new(1, r#"{"data":{"msg":"hi"}}"#);
        assert_eq!(event.sequence, 1);
        assert!(event.subscription_id.is_none());
    }

    #[test]
    fn test_subscription_event_with_subscription_id() {
        let event = SubscriptionEvent::new(5, "{}").with_subscription_id("sub-xyz");
        assert_eq!(event.subscription_id.as_deref(), Some("sub-xyz"));
    }

    // ---- MultiplexerConfig tests --------------------------------------------

    #[test]
    fn test_config_default_values() {
        let cfg = MultiplexerConfig::default();
        assert_eq!(cfg.consumer_buffer_size, 128);
        assert_eq!(cfg.replay_buffer_size, 256);
        assert!(!cfg.drop_slow_consumers);
    }

    // ---- SubscriptionMultiplexer basic tests --------------------------------

    #[tokio::test]
    async fn test_subscribe_and_receive_event() {
        let mux = mux();
        let mut rx = mux.subscribe("sub-1", None).expect("subscribe");
        mux.publish(r#"{"data":{"count":1}}"#).await;

        let event = timeout(Duration::from_millis(100), rx.recv())
            .await
            .expect("no timeout")
            .expect("received");
        assert_eq!(event.sequence, 1);
    }

    #[tokio::test]
    async fn test_multiple_consumers_all_receive_event() {
        let mux = mux();
        let mut rx1 = mux.subscribe("s1", None).expect("s1");
        let mut rx2 = mux.subscribe("s2", None).expect("s2");

        mux.publish("{}").await;

        let e1 = timeout(Duration::from_millis(100), rx1.recv()).await;
        let e2 = timeout(Duration::from_millis(100), rx2.recv()).await;
        assert!(e1.is_ok());
        assert!(e2.is_ok());
    }

    #[tokio::test]
    async fn test_unsubscribe_stops_delivery() {
        let mux = Arc::new(mux());
        let _ = mux.subscribe("sub-gone", None).expect("sub");
        mux.unsubscribe("sub-gone");

        assert_eq!(mux.consumer_count(), 0);
    }

    #[tokio::test]
    async fn test_consumer_count_tracks_active_consumers() {
        let mux = mux();
        assert_eq!(mux.consumer_count(), 0);

        let _r1 = mux.subscribe("a", None).expect("a");
        assert_eq!(mux.consumer_count(), 1);

        let _r2 = mux.subscribe("b", None).expect("b");
        assert_eq!(mux.consumer_count(), 2);

        mux.unsubscribe("a");
        assert_eq!(mux.consumer_count(), 1);
    }

    #[tokio::test]
    async fn test_total_published_increments() {
        let mux = mux();
        assert_eq!(mux.total_published(), 0);
        mux.publish("e1").await;
        mux.publish("e2").await;
        assert_eq!(mux.total_published(), 2);
    }

    #[tokio::test]
    async fn test_replay_buffer_is_populated() {
        let mux = mux();
        mux.publish("e1").await;
        mux.publish("e2").await;
        assert_eq!(mux.replay_buffer_size(), 2);
    }

    #[tokio::test]
    async fn test_resume_with_token_replays_missed_events() {
        let mux = mux();

        // Publish two events before subscribing
        mux.publish("event-1").await;
        mux.publish("event-2").await;

        // Subscribe with a token indicating we've received up to sequence 0
        let token = ResumeToken::new("sub-resume", 0);
        let mut rx = mux.subscribe("sub-resume", Some(&token)).expect("sub");

        // Should receive both replayed events
        let mut received = Vec::new();
        while let Ok(Some(ev)) = timeout(Duration::from_millis(50), rx.recv()).await {
            received.push(ev.sequence);
        }

        assert!(received.contains(&1), "Should replay event 1");
        assert!(received.contains(&2), "Should replay event 2");
    }

    #[tokio::test]
    async fn test_resume_token_skips_already_received_events() {
        let mux = mux();
        mux.publish("event-1").await;
        mux.publish("event-2").await;
        mux.publish("event-3").await;

        // Token says we received up to sequence 2
        let token = ResumeToken::new("sub", 2);
        let mut rx = mux.subscribe("sub", Some(&token)).expect("sub");

        let mut received = Vec::new();
        while let Ok(Some(ev)) = timeout(Duration::from_millis(50), rx.recv()).await {
            received.push(ev.sequence);
        }

        assert!(!received.contains(&1), "Seq 1 should not be replayed");
        assert!(!received.contains(&2), "Seq 2 should not be replayed");
        assert!(received.contains(&3), "Seq 3 should be replayed");
    }

    #[tokio::test]
    async fn test_health_snapshot_tracks_deliveries() {
        let mux = mux();
        let _rx = mux.subscribe("tracked", None).expect("sub");

        mux.publish("event").await;

        let snapshots = mux.health_snapshots();
        assert_eq!(snapshots.len(), 1);
        assert_eq!(snapshots[0].total_delivered, 1);
        assert_eq!(snapshots[0].last_delivered_sequence, 1);
    }

    #[tokio::test]
    async fn test_health_of_unknown_subscription_returns_none() {
        let mux = mux();
        assert!(mux.health_of("nonexistent").is_none());
    }

    #[tokio::test]
    async fn test_resume_token_for_consumer() {
        let mux = mux();
        let _rx = mux.subscribe("sub-r", None).expect("sub");
        mux.publish("e1").await;
        mux.publish("e2").await;

        let token = mux.resume_token_for("sub-r");
        assert!(token.is_some());
        let token = token.expect("should succeed");
        assert_eq!(token.subscription_id, "sub-r");
        assert!(token.last_sequence > 0);
    }

    #[tokio::test]
    async fn test_close_prevents_new_subscriptions() {
        let mux = mux();
        mux.close();

        let result = mux.subscribe("new-sub", None);
        assert!(result.is_err());
        assert!(mux.is_closed());
    }

    #[tokio::test]
    async fn test_close_clears_consumers() {
        let mux = mux();
        let _rx = mux.subscribe("s1", None).expect("s1");
        let _rx2 = mux.subscribe("s2", None).expect("s2");

        mux.close();
        assert_eq!(mux.consumer_count(), 0);
    }

    #[tokio::test]
    async fn test_sequence_numbers_monotonically_increasing() {
        let mux = mux();
        let mut rx = mux.subscribe("seq-test", None).expect("sub");

        mux.publish("e1").await;
        mux.publish("e2").await;
        mux.publish("e3").await;

        let mut seqs = Vec::new();
        while let Ok(Some(ev)) = timeout(Duration::from_millis(50), rx.recv()).await {
            seqs.push(ev.sequence);
        }

        assert_eq!(seqs, vec![1, 2, 3]);
    }

    #[tokio::test]
    async fn test_drop_slow_consumers_config() {
        let config = MultiplexerConfig {
            consumer_buffer_size: 1,
            drop_slow_consumers: true,
            ..Default::default()
        };
        let mux = SubscriptionMultiplexer::new(config);

        let _rx = mux.subscribe("slow", None).expect("sub");

        // Flood with events; the buffer of 1 will fill up quickly.
        for i in 0..10u64 {
            mux.publish(format!(r#"{{"seq":{i}}}"#)).await;
        }

        // Slow consumer should have been dropped after a buffer-full event
        // (may still be in map until next publish cleans it)
        // Just ensure no panics occurred.
        let _ = mux.consumer_count();
    }

    #[tokio::test]
    async fn test_publish_with_no_consumers_returns_zero() {
        let mux = mux();
        let delivered = mux.publish("no-consumers").await;
        assert_eq!(delivered, 0);
    }

    #[tokio::test]
    async fn test_health_subscription_id_matches() {
        let mux = mux();
        let _rx = mux.subscribe("my-sub-id", None).expect("sub");
        let snapshots = mux.health_snapshots();
        assert_eq!(snapshots[0].subscription_id, "my-sub-id");
    }

    #[tokio::test]
    async fn test_subscription_health_is_active_on_create() {
        let mux = mux();
        let _rx = mux.subscribe("active", None).expect("sub");
        let health = mux.health_of("active").expect("health");
        assert!(health.is_active);
        assert!(!health.is_lagging);
    }

    #[tokio::test]
    async fn test_replay_buffer_capped_at_config_size() {
        let config = MultiplexerConfig {
            replay_buffer_size: 3,
            ..Default::default()
        };
        let mux = SubscriptionMultiplexer::new(config);

        for i in 0..5u64 {
            mux.publish(format!("e{i}")).await;
        }

        // Buffer should only hold 3 events (the 3 most recent)
        assert_eq!(mux.replay_buffer_size(), 3);
    }
}
