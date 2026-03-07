//! WebSub (W3C Recommendation) implementation for RDF dataset change notifications.
//!
//! Spec: <https://www.w3.org/TR/websub/>
//!
//! WebSub enables publish-subscribe over HTTP for any web resource. This module provides:
//! - [`WebSubHub`]: Tracks subscriptions and dispatches notifications.
//! - [`WebSubPublisher`]: Notifies a hub that a topic has been updated.
//! - [`WebSubSubscriber`]: Handles intent verification and signature checking.
//! - [`DatasetEventBus`]: In-process change notifications for RDF datasets.

use hmac::{Hmac, Mac};
use sha2::Sha256;
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};
use uuid::Uuid;

// ── Error type ────────────────────────────────────────────────────────────────

/// Errors produced by WebSub operations.
#[derive(Debug)]
pub enum WebSubError {
    /// The hub or callback URL is invalid.
    InvalidUrl(String),
    /// A required parameter is missing.
    MissingParameter(String),
    /// The challenge/signature did not match.
    VerificationFailed(String),
    /// HMAC MAC initialisation failed.
    HmacError(String),
    /// A subscriber callback produced an error.
    SubscriberError(String),
    /// Lock poisoning.
    LockPoisoned,
}

impl fmt::Display for WebSubError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WebSubError::InvalidUrl(msg) => write!(f, "invalid URL: {msg}"),
            WebSubError::MissingParameter(p) => write!(f, "missing parameter: {p}"),
            WebSubError::VerificationFailed(msg) => write!(f, "verification failed: {msg}"),
            WebSubError::HmacError(msg) => write!(f, "HMAC error: {msg}"),
            WebSubError::SubscriberError(msg) => write!(f, "subscriber error: {msg}"),
            WebSubError::LockPoisoned => write!(f, "mutex lock was poisoned"),
        }
    }
}

impl std::error::Error for WebSubError {}

/// Convenience result alias.
pub type Result<T> = std::result::Result<T, WebSubError>;

// ── Subscription ──────────────────────────────────────────────────────────────

/// A single WebSub subscription held by a hub.
#[derive(Debug, Clone)]
pub struct Subscription {
    /// The RDF dataset URL being watched.
    pub topic_url: String,
    /// The subscriber's callback endpoint.
    pub callback_url: String,
    /// Optional HMAC-SHA256 secret shared with the subscriber.
    pub secret: Option<String>,
    /// Requested subscription duration in seconds.
    pub lease_seconds: u64,
    /// Wall-clock time at which this subscription expires.
    pub expires_at: SystemTime,
    /// Whether the hub has confirmed this subscription is active.
    pub active: bool,
}

impl Subscription {
    /// Create a new subscription, computing `expires_at` from `lease_seconds`.
    pub fn new(
        topic_url: impl Into<String>,
        callback_url: impl Into<String>,
        secret: Option<String>,
        lease_seconds: u64,
    ) -> Self {
        let expires_at = SystemTime::now() + Duration::from_secs(lease_seconds);
        Self {
            topic_url: topic_url.into(),
            callback_url: callback_url.into(),
            secret,
            lease_seconds,
            expires_at,
            active: false,
        }
    }

    /// Return `true` if this subscription has not yet expired and is active.
    pub fn is_valid(&self) -> bool {
        self.active && SystemTime::now() < self.expires_at
    }
}

// ── WebSubHub ─────────────────────────────────────────────────────────────────

/// A WebSub hub that stores subscriptions and tracks which topics have been
/// updated.  A production hub would also drive HTTP callbacks; this
/// implementation focuses on the subscription registry.
#[derive(Debug, Default)]
pub struct WebSubHub {
    /// The publicly reachable URL of this hub.
    pub hub_url: String,
    /// All subscriptions held by this hub.
    pub subscriptions: Vec<Subscription>,
}

impl WebSubHub {
    /// Create a new hub with the given public URL.
    pub fn new(hub_url: impl Into<String>) -> Self {
        Self {
            hub_url: hub_url.into(),
            subscriptions: Vec::new(),
        }
    }

    /// Register a new subscription.  Returns the index of the new entry.
    pub fn add_subscription(&mut self, sub: Subscription) -> usize {
        let idx = self.subscriptions.len();
        self.subscriptions.push(sub);
        idx
    }

    /// Activate a subscription by index (after intent verification succeeds).
    pub fn activate(&mut self, index: usize) -> Result<()> {
        let sub = self
            .subscriptions
            .get_mut(index)
            .ok_or_else(|| WebSubError::MissingParameter(format!("index {index}")))?;
        sub.active = true;
        Ok(())
    }

    /// Return all active, non-expired subscriptions for the given topic.
    pub fn active_subscriptions_for(&self, topic_url: &str) -> Vec<&Subscription> {
        self.subscriptions
            .iter()
            .filter(|s| s.topic_url == topic_url && s.is_valid())
            .collect()
    }

    /// Remove expired subscriptions and return the count removed.
    pub fn purge_expired(&mut self) -> usize {
        let before = self.subscriptions.len();
        self.subscriptions
            .retain(|s| SystemTime::now() < s.expires_at);
        before - self.subscriptions.len()
    }
}

// ── WebSubPublisher ───────────────────────────────────────────────────────────

/// Publishes topic updates to a WebSub hub and generates HTTP `Link` headers
/// for self-discovery (RFC 5988 / RFC 8288).
#[derive(Debug, Clone)]
pub struct WebSubPublisher {
    /// The hub to notify.
    pub hub_url: String,
}

impl WebSubPublisher {
    /// Create a publisher that notifies the given hub.
    pub fn new(hub_url: impl Into<String>) -> Self {
        Self {
            hub_url: hub_url.into(),
        }
    }

    /// Build the form-encoded body for a hub ping (`hub.mode=publish`).
    pub fn publish_params(&self, topic_url: &str) -> Vec<(String, String)> {
        vec![
            ("hub.mode".to_string(), "publish".to_string()),
            ("hub.url".to_string(), topic_url.to_string()),
        ]
    }

    /// Notify the hub that `topic_url` has been updated.
    ///
    /// Returns the form-encoded parameter pairs that would be sent to the hub.
    /// Callers are responsible for making the actual HTTP POST.
    pub fn publish(&self, topic_url: &str) -> Result<Vec<(String, String)>> {
        if topic_url.is_empty() {
            return Err(WebSubError::MissingParameter("topic_url".to_string()));
        }
        Ok(self.publish_params(topic_url))
    }

    /// Return the RFC 8288 `Link` headers a publisher should include in its
    /// HTTP responses to allow subscriber discovery.
    ///
    /// ```text
    /// Link: <https://hub.example.com/>; rel="hub"
    /// Link: <https://publisher.example.com/topic>; rel="self"
    /// ```
    pub fn link_headers(&self, topic_url: &str) -> Vec<String> {
        vec![
            format!("<{}>; rel=\"hub\"", self.hub_url),
            format!("<{}>; rel=\"self\"", topic_url),
        ]
    }
}

// ── WebSubSubscriber ──────────────────────────────────────────────────────────

/// Builds subscription requests and handles incoming WebSub callbacks.
#[derive(Debug, Clone)]
pub struct WebSubSubscriber {
    /// The callback URL this subscriber exposes.
    pub callback_url: String,
    /// Optional HMAC-SHA256 secret for signature verification.
    pub secret: Option<String>,
}

impl WebSubSubscriber {
    /// Create a subscriber with the given callback URL and no secret.
    pub fn new(callback_url: impl Into<String>) -> Self {
        Self {
            callback_url: callback_url.into(),
            secret: None,
        }
    }

    /// Attach an HMAC secret (builder style).
    pub fn with_secret(mut self, secret: impl Into<String>) -> Self {
        self.secret = Some(secret.into());
        self
    }

    /// Return the form-encoded parameters for a `subscribe` request.
    pub fn subscribe_params(
        &self,
        hub_url: &str,
        topic_url: &str,
        lease_seconds: u64,
    ) -> Vec<(String, String)> {
        let mut params = vec![
            ("hub.callback".to_string(), self.callback_url.clone()),
            ("hub.mode".to_string(), "subscribe".to_string()),
            ("hub.topic".to_string(), topic_url.to_string()),
            ("hub.lease_seconds".to_string(), lease_seconds.to_string()),
        ];
        if let Some(ref s) = self.secret {
            params.push(("hub.secret".to_string(), s.clone()));
        }
        params.push(("hub.hub_url".to_string(), hub_url.to_string()));
        params
    }

    /// Return the form-encoded parameters for an `unsubscribe` request.
    pub fn unsubscribe_params(&self, hub_url: &str, topic_url: &str) -> Vec<(String, String)> {
        vec![
            ("hub.callback".to_string(), self.callback_url.clone()),
            ("hub.mode".to_string(), "unsubscribe".to_string()),
            ("hub.topic".to_string(), topic_url.to_string()),
            ("hub.hub_url".to_string(), hub_url.to_string()),
        ]
    }

    /// Handle the hub's intent-verification GET request.
    ///
    /// The WebSub spec requires the subscriber to echo back the `challenge`
    /// when it agrees with the `mode` and `topic`.  Returns `Ok(challenge)`.
    pub fn verify_intent(
        &self,
        mode: &str,
        topic: &str,
        challenge: &str,
        _lease_seconds: Option<u64>,
    ) -> Result<String> {
        if mode.is_empty() {
            return Err(WebSubError::MissingParameter("hub.mode".to_string()));
        }
        if topic.is_empty() {
            return Err(WebSubError::MissingParameter("hub.topic".to_string()));
        }
        if challenge.is_empty() {
            return Err(WebSubError::MissingParameter("hub.challenge".to_string()));
        }
        match mode {
            "subscribe" | "unsubscribe" => Ok(challenge.to_string()),
            other => Err(WebSubError::VerificationFailed(format!(
                "unknown mode: {other}"
            ))),
        }
    }

    /// Verify the `X-Hub-Signature` header on an incoming notification.
    ///
    /// The header format is `sha256=<hex-digest>`.  Returns `Ok(true)` on
    /// success, `Ok(false)` if the signature does not match, and `Err` when
    /// no secret is configured or the header is malformed.
    pub fn verify_signature(&self, body: &[u8], signature_header: &str) -> Result<bool> {
        let secret = self
            .secret
            .as_deref()
            .ok_or_else(|| WebSubError::VerificationFailed("no secret configured".to_string()))?;

        let expected_hex = signature_header.strip_prefix("sha256=").ok_or_else(|| {
            WebSubError::VerificationFailed(format!(
                "unsupported signature algorithm in: {signature_header}"
            ))
        })?;

        type HmacSha256 = Hmac<Sha256>;
        let mut mac = HmacSha256::new_from_slice(secret.as_bytes())
            .map_err(|e| WebSubError::HmacError(e.to_string()))?;
        mac.update(body);
        let computed = mac.finalize().into_bytes();
        let computed_hex = hex::encode(computed);

        Ok(computed_hex == expected_hex)
    }

    /// Compute the `X-Hub-Signature` header value for a notification body.
    ///
    /// Useful for hub implementations that need to sign outgoing content.
    pub fn compute_signature(&self, body: &[u8]) -> Result<String> {
        let secret = self
            .secret
            .as_deref()
            .ok_or_else(|| WebSubError::VerificationFailed("no secret configured".to_string()))?;

        type HmacSha256 = Hmac<Sha256>;
        let mut mac = HmacSha256::new_from_slice(secret.as_bytes())
            .map_err(|e| WebSubError::HmacError(e.to_string()))?;
        mac.update(body);
        let digest = mac.finalize().into_bytes();
        Ok(format!("sha256={}", hex::encode(digest)))
    }
}

// ── ChangeType ────────────────────────────────────────────────────────────────

/// The kind of mutation that occurred in an RDF dataset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChangeType {
    /// A single triple was added.
    TripleAdded,
    /// A single triple was removed.
    TripleRemoved,
    /// All triples in a named graph were deleted.
    GraphCleared,
    /// A named graph was dropped entirely.
    GraphDropped,
    /// A new named graph was created (possibly empty).
    GraphCreated,
    /// A bulk transaction affecting many triples at once.
    BulkUpdate {
        /// Number of triples affected by this bulk operation.
        triple_count: usize,
    },
}

// ── DatasetChangeEvent ────────────────────────────────────────────────────────

/// A single change notification produced when an RDF dataset is mutated.
#[derive(Debug, Clone)]
pub struct DatasetChangeEvent {
    /// The dataset that changed.
    pub dataset_url: String,
    /// What kind of change occurred.
    pub change_type: ChangeType,
    /// The named graph affected (if applicable).
    pub affected_graph: Option<String>,
    /// Triples that were added (subject, predicate, object as strings).
    pub added_triples: Vec<(String, String, String)>,
    /// Triples that were removed (subject, predicate, object as strings).
    pub removed_triples: Vec<(String, String, String)>,
    /// When the change occurred.
    pub timestamp: SystemTime,
    /// Unique identifier for deduplication across distributed systems.
    pub change_id: String,
}

impl DatasetChangeEvent {
    /// Construct a new change event, generating a fresh UUID for `change_id`.
    pub fn new(
        dataset_url: impl Into<String>,
        change_type: ChangeType,
        affected_graph: Option<String>,
        added_triples: Vec<(String, String, String)>,
        removed_triples: Vec<(String, String, String)>,
    ) -> Self {
        Self {
            dataset_url: dataset_url.into(),
            change_type,
            affected_graph,
            added_triples,
            removed_triples,
            timestamp: SystemTime::now(),
            change_id: Uuid::new_v4().to_string(),
        }
    }

    /// Return `true` if this event contains at least one triple mutation.
    pub fn has_delta(&self) -> bool {
        !self.added_triples.is_empty() || !self.removed_triples.is_empty()
    }
}

// ── ChangeSubscriber trait ────────────────────────────────────────────────────

/// Implemented by anything that wants to receive dataset change notifications.
pub trait ChangeSubscriber: Send + Sync {
    /// Called synchronously on the publishing thread.
    fn on_change(&self, event: &DatasetChangeEvent) -> Result<()>;
}

// ── DatasetEventBus ───────────────────────────────────────────────────────────

/// An in-process publish-subscribe bus for RDF dataset change events.
///
/// Multiple [`ChangeSubscriber`] implementations can be registered.  When
/// [`DatasetEventBus::publish`] is called every subscriber receives the event
/// in registration order.  Errors from individual subscribers are collected
/// but do not prevent later subscribers from receiving the event.
pub struct DatasetEventBus {
    subscribers: Vec<Box<dyn ChangeSubscriber>>,
}

impl Default for DatasetEventBus {
    fn default() -> Self {
        Self::new()
    }
}

impl DatasetEventBus {
    /// Create a new, empty event bus.
    pub fn new() -> Self {
        Self {
            subscribers: Vec::new(),
        }
    }

    /// Register a subscriber.
    pub fn subscribe(&mut self, subscriber: Box<dyn ChangeSubscriber>) {
        self.subscribers.push(subscriber);
    }

    /// Publish a change event to all registered subscribers.
    ///
    /// All subscribers are called even if earlier ones return an error.
    /// If any subscriber failed the first error is returned.
    pub fn publish(&self, event: DatasetChangeEvent) -> Result<()> {
        let mut first_error: Option<WebSubError> = None;
        for sub in &self.subscribers {
            if let Err(e) = sub.on_change(&event) {
                if first_error.is_none() {
                    first_error = Some(e);
                }
            }
        }
        match first_error {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// Return the number of registered subscribers.
    pub fn subscriber_count(&self) -> usize {
        self.subscribers.len()
    }
}

// ── InMemorySubscriber ────────────────────────────────────────────────────────

/// A [`ChangeSubscriber`] that collects received events in a `Vec` behind a
/// shared `Mutex`.  Useful for unit tests.
pub struct InMemorySubscriber {
    received: Arc<Mutex<Vec<DatasetChangeEvent>>>,
}

impl InMemorySubscriber {
    /// Create a new subscriber and return it together with a shared handle to
    /// the underlying event buffer so tests can inspect received events without
    /// keeping a reference to the subscriber itself.
    pub fn new() -> (Self, Arc<Mutex<Vec<DatasetChangeEvent>>>) {
        let received = Arc::new(Mutex::new(Vec::new()));
        let handle = Arc::clone(&received);
        (Self { received }, handle)
    }

    /// Return a snapshot of all received events (cloned).
    pub fn events(&self) -> Vec<DatasetChangeEvent> {
        self.received.lock().map(|g| g.clone()).unwrap_or_default()
    }
}

impl Default for InMemorySubscriber {
    fn default() -> Self {
        Self {
            received: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

impl ChangeSubscriber for InMemorySubscriber {
    fn on_change(&self, event: &DatasetChangeEvent) -> Result<()> {
        self.received
            .lock()
            .map_err(|_| WebSubError::LockPoisoned)?
            .push(event.clone());
        Ok(())
    }
}

// ── Helper: compute HMAC-SHA256 signature ─────────────────────────────────────

/// Compute a raw HMAC-SHA256 hex digest over `body` using `secret`.
///
/// Returns a `sha256=<hex>` formatted string suitable for the
/// `X-Hub-Signature` header.
pub fn sign_notification(secret: &str, body: &[u8]) -> Result<String> {
    type HmacSha256 = Hmac<Sha256>;
    let mut mac = HmacSha256::new_from_slice(secret.as_bytes())
        .map_err(|e| WebSubError::HmacError(e.to_string()))?;
    mac.update(body);
    let digest = mac.finalize().into_bytes();
    Ok(format!("sha256={}", hex::encode(digest)))
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn make_publisher() -> WebSubPublisher {
        WebSubPublisher::new("https://hub.example.com/")
    }

    fn make_subscriber() -> WebSubSubscriber {
        WebSubSubscriber::new("https://subscriber.example.com/callback")
    }

    fn make_secret_subscriber() -> WebSubSubscriber {
        WebSubSubscriber::new("https://subscriber.example.com/callback")
            .with_secret("my-secret-key")
    }

    fn make_event(ct: ChangeType) -> DatasetChangeEvent {
        DatasetChangeEvent::new("https://ds.example.com/", ct, None, vec![], vec![])
    }

    // ── WebSubPublisher ───────────────────────────────────────────────────────

    #[test]
    fn publisher_link_headers_hub_rel() {
        let pub_ = make_publisher();
        let headers = pub_.link_headers("https://publisher.example.com/topic");
        assert_eq!(headers.len(), 2);
        assert!(
            headers[0].contains("rel=\"hub\""),
            "first header should have rel=hub"
        );
    }

    #[test]
    fn publisher_link_headers_self_rel() {
        let pub_ = make_publisher();
        let headers = pub_.link_headers("https://publisher.example.com/topic");
        assert!(
            headers[1].contains("rel=\"self\""),
            "second header should have rel=self"
        );
    }

    #[test]
    fn publisher_link_headers_contain_hub_url() {
        let pub_ = make_publisher();
        let headers = pub_.link_headers("https://publisher.example.com/topic");
        assert!(headers[0].contains("https://hub.example.com/"));
    }

    #[test]
    fn publisher_link_headers_contain_topic_url() {
        let pub_ = make_publisher();
        let topic = "https://publisher.example.com/topic";
        let headers = pub_.link_headers(topic);
        assert!(headers[1].contains(topic));
    }

    #[test]
    fn publisher_publish_returns_params() {
        let pub_ = make_publisher();
        let params = pub_.publish("https://ds.example.com/").unwrap();
        assert!(params
            .iter()
            .any(|(k, v)| k == "hub.mode" && v == "publish"));
    }

    #[test]
    fn publisher_publish_includes_topic() {
        let pub_ = make_publisher();
        let topic = "https://ds.example.com/dataset1";
        let params = pub_.publish(topic).unwrap();
        assert!(params.iter().any(|(k, v)| k == "hub.url" && v == topic));
    }

    #[test]
    fn publisher_publish_empty_topic_error() {
        let pub_ = make_publisher();
        assert!(pub_.publish("").is_err());
    }

    #[test]
    fn publisher_new_stores_hub_url() {
        let pub_ = WebSubPublisher::new("https://custom.hub.com/");
        assert_eq!(pub_.hub_url, "https://custom.hub.com/");
    }

    // ── WebSubSubscriber: subscribe_params ─────────────────────────────────

    #[test]
    fn subscriber_subscribe_params_mode() {
        let sub = make_subscriber();
        let params = sub.subscribe_params(
            "https://hub.example.com/",
            "https://topic.example.com/",
            86400,
        );
        assert!(params
            .iter()
            .any(|(k, v)| k == "hub.mode" && v == "subscribe"));
    }

    #[test]
    fn subscriber_subscribe_params_callback() {
        let sub = make_subscriber();
        let params = sub.subscribe_params(
            "https://hub.example.com/",
            "https://topic.example.com/",
            86400,
        );
        assert!(params
            .iter()
            .any(|(k, v)| k == "hub.callback" && v == "https://subscriber.example.com/callback"));
    }

    #[test]
    fn subscriber_subscribe_params_topic() {
        let sub = make_subscriber();
        let topic = "https://topic.example.com/graph1";
        let params = sub.subscribe_params("https://hub.example.com/", topic, 86400);
        assert!(params.iter().any(|(k, v)| k == "hub.topic" && v == topic));
    }

    #[test]
    fn subscriber_subscribe_params_lease_seconds() {
        let sub = make_subscriber();
        let params = sub.subscribe_params(
            "https://hub.example.com/",
            "https://topic.example.com/",
            3600,
        );
        assert!(params
            .iter()
            .any(|(k, v)| k == "hub.lease_seconds" && v == "3600"));
    }

    #[test]
    fn subscriber_subscribe_params_with_secret() {
        let sub = make_secret_subscriber();
        let params = sub.subscribe_params(
            "https://hub.example.com/",
            "https://topic.example.com/",
            86400,
        );
        assert!(params
            .iter()
            .any(|(k, v)| k == "hub.secret" && v == "my-secret-key"));
    }

    #[test]
    fn subscriber_subscribe_params_no_secret_if_unset() {
        let sub = make_subscriber();
        let params = sub.subscribe_params(
            "https://hub.example.com/",
            "https://topic.example.com/",
            86400,
        );
        assert!(!params.iter().any(|(k, _)| k == "hub.secret"));
    }

    // ── WebSubSubscriber: unsubscribe_params ──────────────────────────────

    #[test]
    fn subscriber_unsubscribe_params_mode() {
        let sub = make_subscriber();
        let params =
            sub.unsubscribe_params("https://hub.example.com/", "https://topic.example.com/");
        assert!(params
            .iter()
            .any(|(k, v)| k == "hub.mode" && v == "unsubscribe"));
    }

    #[test]
    fn subscriber_unsubscribe_params_topic() {
        let sub = make_subscriber();
        let topic = "https://topic.example.com/";
        let params = sub.unsubscribe_params("https://hub.example.com/", topic);
        assert!(params.iter().any(|(k, v)| k == "hub.topic" && v == topic));
    }

    #[test]
    fn subscriber_unsubscribe_params_callback() {
        let sub = make_subscriber();
        let params =
            sub.unsubscribe_params("https://hub.example.com/", "https://topic.example.com/");
        assert!(params.iter().any(|(k, _)| k == "hub.callback"));
    }

    // ── WebSubSubscriber: verify_intent ───────────────────────────────────

    #[test]
    fn verify_intent_subscribe_returns_challenge() {
        let sub = make_subscriber();
        let challenge = "abc123xyz";
        let result = sub.verify_intent(
            "subscribe",
            "https://topic.example.com/",
            challenge,
            Some(86400),
        );
        assert_eq!(result.unwrap(), challenge);
    }

    #[test]
    fn verify_intent_unsubscribe_returns_challenge() {
        let sub = make_subscriber();
        let challenge = "unsubchallenge";
        let result =
            sub.verify_intent("unsubscribe", "https://topic.example.com/", challenge, None);
        assert_eq!(result.unwrap(), challenge);
    }

    #[test]
    fn verify_intent_unknown_mode_is_error() {
        let sub = make_subscriber();
        let result = sub.verify_intent("denied", "https://topic.example.com/", "ch", None);
        assert!(result.is_err());
    }

    #[test]
    fn verify_intent_empty_mode_is_error() {
        let sub = make_subscriber();
        let result = sub.verify_intent("", "https://topic.example.com/", "ch", None);
        assert!(result.is_err());
    }

    #[test]
    fn verify_intent_empty_topic_is_error() {
        let sub = make_subscriber();
        let result = sub.verify_intent("subscribe", "", "ch", None);
        assert!(result.is_err());
    }

    #[test]
    fn verify_intent_empty_challenge_is_error() {
        let sub = make_subscriber();
        let result = sub.verify_intent("subscribe", "https://topic.example.com/", "", None);
        assert!(result.is_err());
    }

    // ── WebSubSubscriber: verify_signature ────────────────────────────────

    #[test]
    fn verify_signature_valid() {
        let sub = make_secret_subscriber();
        let body = b"hello world notification body";
        let sig = sub.compute_signature(body).unwrap();
        assert!(sub.verify_signature(body, &sig).unwrap());
    }

    #[test]
    fn verify_signature_wrong_secret_returns_false() {
        let sub_signer =
            WebSubSubscriber::new("https://sub.example.com/cb").with_secret("correct-secret");
        let sub_verifier =
            WebSubSubscriber::new("https://sub.example.com/cb").with_secret("wrong-secret");
        let body = b"notification payload";
        let sig = sub_signer.compute_signature(body).unwrap();
        assert!(!sub_verifier.verify_signature(body, &sig).unwrap());
    }

    #[test]
    fn verify_signature_tampered_body_returns_false() {
        let sub = make_secret_subscriber();
        let body = b"original body";
        let sig = sub.compute_signature(body).unwrap();
        assert!(!sub.verify_signature(b"tampered body", &sig).unwrap());
    }

    #[test]
    fn verify_signature_no_secret_is_error() {
        let sub = make_subscriber();
        let result = sub.verify_signature(b"body", "sha256=abc");
        assert!(result.is_err());
    }

    #[test]
    fn verify_signature_bad_algorithm_prefix_is_error() {
        let sub = make_secret_subscriber();
        let result = sub.verify_signature(b"body", "md5=deadbeef");
        assert!(result.is_err());
    }

    #[test]
    fn compute_signature_format_starts_with_sha256() {
        let sub = make_secret_subscriber();
        let sig = sub.compute_signature(b"data").unwrap();
        assert!(sig.starts_with("sha256="));
    }

    #[test]
    fn compute_signature_hex_length() {
        // SHA-256 produces 32 bytes => 64 hex chars + "sha256=" prefix = 71
        let sub = make_secret_subscriber();
        let sig = sub.compute_signature(b"data").unwrap();
        assert_eq!(sig.len(), 71);
    }

    // ── sign_notification helper ──────────────────────────────────────────

    #[test]
    fn sign_notification_helper_matches_subscriber() {
        let secret = "shared-secret";
        let body = b"rdf patch notification";
        let sig = sign_notification(secret, body).unwrap();
        let sub = WebSubSubscriber::new("cb").with_secret(secret);
        assert!(sub.verify_signature(body, &sig).unwrap());
    }

    // ── ChangeType variants ───────────────────────────────────────────────

    #[test]
    fn change_type_triple_added() {
        let ct = ChangeType::TripleAdded;
        assert_eq!(ct, ChangeType::TripleAdded);
    }

    #[test]
    fn change_type_triple_removed() {
        let ct = ChangeType::TripleRemoved;
        assert_eq!(ct, ChangeType::TripleRemoved);
    }

    #[test]
    fn change_type_graph_cleared() {
        let ct = ChangeType::GraphCleared;
        assert_eq!(ct, ChangeType::GraphCleared);
    }

    #[test]
    fn change_type_graph_dropped() {
        let ct = ChangeType::GraphDropped;
        assert_eq!(ct, ChangeType::GraphDropped);
    }

    #[test]
    fn change_type_graph_created() {
        let ct = ChangeType::GraphCreated;
        assert_eq!(ct, ChangeType::GraphCreated);
    }

    #[test]
    fn change_type_bulk_update_with_count() {
        let ct = ChangeType::BulkUpdate { triple_count: 500 };
        if let ChangeType::BulkUpdate { triple_count } = ct {
            assert_eq!(triple_count, 500);
        } else {
            panic!("expected BulkUpdate");
        }
    }

    #[test]
    fn change_type_bulk_update_zero_count() {
        let ct = ChangeType::BulkUpdate { triple_count: 0 };
        if let ChangeType::BulkUpdate { triple_count } = ct {
            assert_eq!(triple_count, 0);
        } else {
            panic!("expected BulkUpdate");
        }
    }

    // ── DatasetChangeEvent ────────────────────────────────────────────────

    #[test]
    fn dataset_change_event_has_change_id() {
        let ev = make_event(ChangeType::TripleAdded);
        assert!(!ev.change_id.is_empty());
    }

    #[test]
    fn dataset_change_event_unique_ids() {
        let ev1 = make_event(ChangeType::TripleAdded);
        let ev2 = make_event(ChangeType::TripleAdded);
        assert_ne!(ev1.change_id, ev2.change_id);
    }

    #[test]
    fn dataset_change_event_has_delta_true_when_triples() {
        let ev = DatasetChangeEvent::new(
            "https://ds.example.com/",
            ChangeType::TripleAdded,
            None,
            vec![("<s>".to_string(), "<p>".to_string(), "<o>".to_string())],
            vec![],
        );
        assert!(ev.has_delta());
    }

    #[test]
    fn dataset_change_event_has_delta_false_when_empty() {
        let ev = make_event(ChangeType::GraphCreated);
        assert!(!ev.has_delta());
    }

    #[test]
    fn dataset_change_event_stores_dataset_url() {
        let ev = DatasetChangeEvent::new(
            "https://my-dataset.example.com/",
            ChangeType::GraphCreated,
            None,
            vec![],
            vec![],
        );
        assert_eq!(ev.dataset_url, "https://my-dataset.example.com/");
    }

    #[test]
    fn dataset_change_event_stores_affected_graph() {
        let graph = "https://my-dataset.example.com/graph1";
        let ev = DatasetChangeEvent::new(
            "https://my-dataset.example.com/",
            ChangeType::GraphCreated,
            Some(graph.to_string()),
            vec![],
            vec![],
        );
        assert_eq!(ev.affected_graph.as_deref(), Some(graph));
    }

    // ── DatasetEventBus ───────────────────────────────────────────────────

    #[test]
    fn event_bus_subscriber_count_zero_initially() {
        let bus = DatasetEventBus::new();
        assert_eq!(bus.subscriber_count(), 0);
    }

    #[test]
    fn event_bus_subscriber_count_after_register() {
        let mut bus = DatasetEventBus::new();
        let (sub, _) = InMemorySubscriber::new();
        bus.subscribe(Box::new(sub));
        assert_eq!(bus.subscriber_count(), 1);
    }

    #[test]
    fn event_bus_multiple_subscriber_count() {
        let mut bus = DatasetEventBus::new();
        for _ in 0..5 {
            let (sub, _) = InMemorySubscriber::new();
            bus.subscribe(Box::new(sub));
        }
        assert_eq!(bus.subscriber_count(), 5);
    }

    #[test]
    fn event_bus_publish_to_single_subscriber() {
        let mut bus = DatasetEventBus::new();
        let (sub, handle) = InMemorySubscriber::new();
        bus.subscribe(Box::new(sub));

        let ev = make_event(ChangeType::TripleAdded);
        bus.publish(ev).unwrap();

        let events = handle.lock().unwrap();
        assert_eq!(events.len(), 1);
    }

    #[test]
    fn event_bus_publish_to_multiple_subscribers() {
        let mut bus = DatasetEventBus::new();
        let (sub1, handle1) = InMemorySubscriber::new();
        let (sub2, handle2) = InMemorySubscriber::new();
        bus.subscribe(Box::new(sub1));
        bus.subscribe(Box::new(sub2));

        bus.publish(make_event(ChangeType::GraphCreated)).unwrap();

        assert_eq!(handle1.lock().unwrap().len(), 1);
        assert_eq!(handle2.lock().unwrap().len(), 1);
    }

    #[test]
    fn event_bus_publish_preserves_event_data() {
        let mut bus = DatasetEventBus::new();
        let (sub, handle) = InMemorySubscriber::new();
        bus.subscribe(Box::new(sub));

        let ev = DatasetChangeEvent::new(
            "https://ds.example.com/",
            ChangeType::BulkUpdate { triple_count: 42 },
            Some("https://ds.example.com/graph1".to_string()),
            vec![("<s>".to_string(), "<p>".to_string(), "<o>".to_string())],
            vec![],
        );
        let expected_id = ev.change_id.clone();
        bus.publish(ev).unwrap();

        let events = handle.lock().unwrap();
        assert_eq!(events[0].change_id, expected_id);
    }

    // ── InMemorySubscriber ────────────────────────────────────────────────

    #[test]
    fn in_memory_subscriber_receives_event() {
        let (sub, _handle) = InMemorySubscriber::new();
        let ev = make_event(ChangeType::TripleAdded);
        sub.on_change(&ev).unwrap();
        let events = sub.events();
        assert_eq!(events.len(), 1);
    }

    #[test]
    fn in_memory_subscriber_accumulates_multiple_events() {
        let (sub, _handle) = InMemorySubscriber::new();
        for ct in [
            ChangeType::TripleAdded,
            ChangeType::TripleRemoved,
            ChangeType::GraphCleared,
        ] {
            sub.on_change(&make_event(ct)).unwrap();
        }
        assert_eq!(sub.events().len(), 3);
    }

    #[test]
    fn in_memory_subscriber_shared_handle_sees_same_events() {
        let (sub, handle) = InMemorySubscriber::new();
        sub.on_change(&make_event(ChangeType::GraphDropped))
            .unwrap();
        assert_eq!(handle.lock().unwrap().len(), 1);
    }

    #[test]
    fn in_memory_subscriber_event_change_type_preserved() {
        let (sub, _) = InMemorySubscriber::new();
        sub.on_change(&make_event(ChangeType::BulkUpdate { triple_count: 100 }))
            .unwrap();
        let events = sub.events();
        if let ChangeType::BulkUpdate { triple_count } = &events[0].change_type {
            assert_eq!(*triple_count, 100);
        } else {
            panic!("wrong change type");
        }
    }

    // ── Round-trip: publish event -> subscriber receives it ────────────────

    #[test]
    fn round_trip_single_event() {
        let mut bus = DatasetEventBus::new();
        let (sub, handle) = InMemorySubscriber::new();
        bus.subscribe(Box::new(sub));

        let ev = DatasetChangeEvent::new(
            "https://ds.example.com/",
            ChangeType::TripleAdded,
            None,
            vec![(
                "<http://s>".to_string(),
                "<http://p>".to_string(),
                "<http://o>".to_string(),
            )],
            vec![],
        );
        bus.publish(ev).unwrap();

        let guard = handle.lock().unwrap();
        assert_eq!(guard.len(), 1);
        assert!(guard[0].has_delta());
    }

    #[test]
    fn round_trip_bulk_update_event() {
        let mut bus = DatasetEventBus::new();
        let (sub, handle) = InMemorySubscriber::new();
        bus.subscribe(Box::new(sub));

        let ev = DatasetChangeEvent::new(
            "https://ds.example.com/",
            ChangeType::BulkUpdate { triple_count: 9999 },
            None,
            vec![],
            vec![],
        );
        bus.publish(ev).unwrap();

        let guard = handle.lock().unwrap();
        assert_eq!(guard.len(), 1);
        if let ChangeType::BulkUpdate { triple_count } = guard[0].change_type {
            assert_eq!(triple_count, 9999);
        } else {
            panic!("expected BulkUpdate");
        }
    }

    #[test]
    fn round_trip_graph_cleared_event() {
        let mut bus = DatasetEventBus::new();
        let (sub, handle) = InMemorySubscriber::new();
        bus.subscribe(Box::new(sub));

        bus.publish(make_event(ChangeType::GraphCleared)).unwrap();

        let guard = handle.lock().unwrap();
        assert_eq!(guard[0].change_type, ChangeType::GraphCleared);
    }

    // ── WebSubHub ─────────────────────────────────────────────────────────

    #[test]
    fn hub_add_and_count_subscriptions() {
        let mut hub = WebSubHub::new("https://hub.example.com/");
        let sub = Subscription::new(
            "https://topic.example.com/",
            "https://sub.example.com/cb",
            None,
            3600,
        );
        hub.add_subscription(sub);
        assert_eq!(hub.subscriptions.len(), 1);
    }

    #[test]
    fn hub_activate_subscription() {
        let mut hub = WebSubHub::new("https://hub.example.com/");
        let sub = Subscription::new(
            "https://topic.example.com/",
            "https://sub.example.com/cb",
            None,
            3600,
        );
        let idx = hub.add_subscription(sub);
        hub.activate(idx).unwrap();
        assert!(hub.subscriptions[idx].active);
    }

    #[test]
    fn hub_active_subscriptions_for_topic() {
        let mut hub = WebSubHub::new("https://hub.example.com/");
        let sub = Subscription::new(
            "https://topic.example.com/",
            "https://sub.example.com/cb",
            None,
            3600,
        );
        let idx = hub.add_subscription(sub);
        hub.activate(idx).unwrap();
        let active = hub.active_subscriptions_for("https://topic.example.com/");
        assert_eq!(active.len(), 1);
    }

    #[test]
    fn hub_active_subscriptions_empty_for_unknown_topic() {
        let hub = WebSubHub::new("https://hub.example.com/");
        let active = hub.active_subscriptions_for("https://unknown.example.com/");
        assert!(active.is_empty());
    }

    #[test]
    fn subscription_new_is_not_active_by_default() {
        let sub = Subscription::new(
            "https://topic.example.com/",
            "https://cb.example.com/",
            None,
            3600,
        );
        assert!(!sub.active);
    }

    #[test]
    fn subscription_is_valid_when_active_and_not_expired() {
        let mut sub = Subscription::new(
            "https://topic.example.com/",
            "https://cb.example.com/",
            None,
            9999,
        );
        sub.active = true;
        assert!(sub.is_valid());
    }

    #[test]
    fn subscription_is_invalid_when_not_active() {
        let sub = Subscription::new(
            "https://topic.example.com/",
            "https://cb.example.com/",
            None,
            9999,
        );
        assert!(!sub.is_valid());
    }
}
