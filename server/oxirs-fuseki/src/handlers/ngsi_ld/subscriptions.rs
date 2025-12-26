//! NGSI-LD Subscription Operations
//!
//! Implements subscription management for entity change notifications.

use super::content_neg::NgsiContentNegotiator;
use super::types::{NgsiError, NgsiSubscription, SubscriptionStatus, SubscriptionType};
use axum::{
    body::Bytes,
    extract::{Path, Query},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use chrono::Utc;
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Subscription store trait
#[async_trait::async_trait]
pub trait SubscriptionStore: Send + Sync {
    /// Get subscription by ID
    async fn get_subscription(&self, id: &str) -> Result<Option<NgsiSubscription>, NgsiError>;

    /// Store subscription
    async fn store_subscription(&self, subscription: &NgsiSubscription) -> Result<(), NgsiError>;

    /// Update subscription
    async fn update_subscription(&self, subscription: &NgsiSubscription) -> Result<(), NgsiError>;

    /// Delete subscription
    async fn delete_subscription(&self, id: &str) -> Result<(), NgsiError>;

    /// List subscriptions
    async fn list_subscriptions(
        &self,
        limit: Option<u32>,
        offset: Option<u32>,
    ) -> Result<Vec<NgsiSubscription>, NgsiError>;

    /// Count subscriptions
    async fn count_subscriptions(&self) -> Result<usize, NgsiError>;
}

/// In-memory subscription store (for testing and development)
pub struct InMemorySubscriptionStore {
    subscriptions: RwLock<HashMap<String, NgsiSubscription>>,
}

impl Default for InMemorySubscriptionStore {
    fn default() -> Self {
        Self::new()
    }
}

impl InMemorySubscriptionStore {
    pub fn new() -> Self {
        Self {
            subscriptions: RwLock::new(HashMap::new()),
        }
    }
}

#[async_trait::async_trait]
impl SubscriptionStore for InMemorySubscriptionStore {
    async fn get_subscription(&self, id: &str) -> Result<Option<NgsiSubscription>, NgsiError> {
        Ok(self.subscriptions.read().await.get(id).cloned())
    }

    async fn store_subscription(&self, subscription: &NgsiSubscription) -> Result<(), NgsiError> {
        if let Some(ref id) = subscription.id {
            self.subscriptions
                .write()
                .await
                .insert(id.clone(), subscription.clone());
        }
        Ok(())
    }

    async fn update_subscription(&self, subscription: &NgsiSubscription) -> Result<(), NgsiError> {
        if let Some(ref id) = subscription.id {
            self.subscriptions
                .write()
                .await
                .insert(id.clone(), subscription.clone());
        }
        Ok(())
    }

    async fn delete_subscription(&self, id: &str) -> Result<(), NgsiError> {
        self.subscriptions.write().await.remove(id);
        Ok(())
    }

    async fn list_subscriptions(
        &self,
        limit: Option<u32>,
        offset: Option<u32>,
    ) -> Result<Vec<NgsiSubscription>, NgsiError> {
        let subs = self.subscriptions.read().await;
        let offset = offset.unwrap_or(0) as usize;
        let limit = limit.unwrap_or(100) as usize;

        Ok(subs.values().skip(offset).take(limit).cloned().collect())
    }

    async fn count_subscriptions(&self) -> Result<usize, NgsiError> {
        Ok(self.subscriptions.read().await.len())
    }
}

/// Query parameters for subscription list
#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SubscriptionQueryParams {
    pub limit: Option<u32>,
    pub offset: Option<u32>,
    pub options: Option<String>,
}

impl SubscriptionQueryParams {
    pub fn is_count(&self) -> bool {
        self.options
            .as_ref()
            .map(|o| o.contains("count"))
            .unwrap_or(false)
    }
}

/// Create subscription
///
/// POST /ngsi-ld/v1/subscriptions
pub async fn create_subscription(
    headers: HeaderMap,
    body: Bytes,
    store: Arc<dyn SubscriptionStore>,
) -> Result<Response, NgsiError> {
    let negotiator = NgsiContentNegotiator::new();

    // Validate content type
    negotiator.validate_content_type(&headers)?;

    // Parse subscription
    let mut subscription: NgsiSubscription = serde_json::from_slice(&body)
        .map_err(|e| NgsiError::InvalidRequest(format!("Invalid subscription JSON: {}", e)))?;

    // Generate ID if not provided
    if subscription.id.is_none() {
        subscription.id = Some(format!("urn:ngsi-ld:Subscription:{}", Uuid::new_v4()));
    }

    // Set type
    subscription.sub_type = SubscriptionType::Subscription;

    // Set timestamps
    let now = Utc::now();
    subscription.created_at = Some(now);
    subscription.modified_at = Some(now);

    // Set initial status
    subscription.status = Some(SubscriptionStatus::Active);
    subscription.is_active = Some(true);

    // Validate subscription
    validate_subscription(&subscription)?;

    // Store subscription
    store.store_subscription(&subscription).await?;

    let id = subscription.id.as_ref().unwrap();

    Ok((
        StatusCode::CREATED,
        [
            ("Location", format!("/ngsi-ld/v1/subscriptions/{}", id)),
            ("Content-Type", "application/json".to_string()),
        ],
    )
        .into_response())
}

/// Get subscription by ID
///
/// GET /ngsi-ld/v1/subscriptions/:id
pub async fn get_subscription(
    Path(subscription_id): Path<String>,
    headers: HeaderMap,
    store: Arc<dyn SubscriptionStore>,
) -> Result<Response, NgsiError> {
    let negotiator = NgsiContentNegotiator::new();
    let format = negotiator.negotiate_response(&headers)?;

    let subscription = store
        .get_subscription(&subscription_id)
        .await?
        .ok_or_else(|| {
            NgsiError::NotFound(format!("Subscription {} not found", subscription_id))
        })?;

    Ok((
        StatusCode::OK,
        [("Content-Type", format.mime_type())],
        Json(subscription),
    )
        .into_response())
}

/// List subscriptions
///
/// GET /ngsi-ld/v1/subscriptions
pub async fn list_subscriptions(
    Query(params): Query<SubscriptionQueryParams>,
    headers: HeaderMap,
    store: Arc<dyn SubscriptionStore>,
) -> Result<Response, NgsiError> {
    let negotiator = NgsiContentNegotiator::new();
    let format = negotiator.negotiate_response(&headers)?;

    let subscriptions = store
        .list_subscriptions(params.limit, params.offset)
        .await?;

    let mut response_headers = vec![("Content-Type", format.mime_type().to_string())];

    if params.is_count() {
        let count = store.count_subscriptions().await?;
        response_headers.push(("NGSILD-Results-Count", count.to_string()));
    }

    use axum::http::header::{HeaderName, HeaderValue};
    let mut response = (StatusCode::OK, Json(subscriptions)).into_response();
    for (key, value) in response_headers {
        if let (Ok(name), Ok(val)) = (key.parse::<HeaderName>(), value.parse::<HeaderValue>()) {
            response.headers_mut().insert(name, val);
        }
    }
    Ok(response)
}

/// Update subscription
///
/// PATCH /ngsi-ld/v1/subscriptions/:id
pub async fn update_subscription(
    Path(subscription_id): Path<String>,
    headers: HeaderMap,
    body: Bytes,
    store: Arc<dyn SubscriptionStore>,
) -> Result<Response, NgsiError> {
    let negotiator = NgsiContentNegotiator::new();

    // Validate content type
    negotiator.validate_content_type(&headers)?;

    // Get existing subscription
    let mut subscription = store
        .get_subscription(&subscription_id)
        .await?
        .ok_or_else(|| {
            NgsiError::NotFound(format!("Subscription {} not found", subscription_id))
        })?;

    // Parse patch
    let patch: serde_json::Value = serde_json::from_slice(&body)
        .map_err(|e| NgsiError::InvalidRequest(format!("Invalid patch JSON: {}", e)))?;

    // Apply patch fields
    if let Some(obj) = patch.as_object() {
        if let Some(name) = obj.get("name").and_then(|v| v.as_str()) {
            subscription.name = Some(name.to_string());
        }
        if let Some(description) = obj.get("description").and_then(|v| v.as_str()) {
            subscription.description = Some(description.to_string());
        }
        if let Some(throttling) = obj.get("throttling").and_then(|v| v.as_u64()) {
            subscription.throttling = Some(throttling);
        }
        if let Some(expires_at) = obj.get("expiresAt").and_then(|v| v.as_str()) {
            subscription.expires_at = chrono::DateTime::parse_from_rfc3339(expires_at)
                .ok()
                .map(|dt| dt.with_timezone(&Utc));
        }
        if let Some(is_active) = obj.get("isActive").and_then(|v| v.as_bool()) {
            subscription.is_active = Some(is_active);
            subscription.status = Some(if is_active {
                SubscriptionStatus::Active
            } else {
                SubscriptionStatus::Paused
            });
        }
    }

    // Update timestamp
    subscription.modified_at = Some(Utc::now());

    // Store updated subscription
    store.update_subscription(&subscription).await?;

    Ok(StatusCode::NO_CONTENT.into_response())
}

/// Delete subscription
///
/// DELETE /ngsi-ld/v1/subscriptions/:id
pub async fn delete_subscription(
    Path(subscription_id): Path<String>,
    store: Arc<dyn SubscriptionStore>,
) -> Result<Response, NgsiError> {
    // Check if subscription exists
    let _ = store
        .get_subscription(&subscription_id)
        .await?
        .ok_or_else(|| {
            NgsiError::NotFound(format!("Subscription {} not found", subscription_id))
        })?;

    // Delete subscription
    store.delete_subscription(&subscription_id).await?;

    Ok(StatusCode::NO_CONTENT.into_response())
}

/// Validate subscription structure
fn validate_subscription(subscription: &NgsiSubscription) -> Result<(), NgsiError> {
    // Check entities selector is present
    if subscription.entities.is_empty() {
        return Err(NgsiError::InvalidRequest(
            "Subscription must have at least one entity selector".to_string(),
        ));
    }

    // Validate each entity selector
    for selector in &subscription.entities {
        if selector.id.is_none() && selector.id_pattern.is_none() && selector.entity_type.is_none()
        {
            return Err(NgsiError::InvalidRequest(
                "Entity selector must have id, idPattern, or type".to_string(),
            ));
        }
    }

    // Validate notification endpoint
    if subscription.notification.endpoint.uri.is_empty() {
        return Err(NgsiError::InvalidRequest(
            "Notification endpoint URI is required".to_string(),
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::types::{EntitySelector, NotificationEndpoint, NotificationParams};
    use super::*;

    fn create_test_subscription() -> NgsiSubscription {
        NgsiSubscription {
            id: Some("urn:ngsi-ld:Subscription:test001".to_string()),
            sub_type: SubscriptionType::Subscription,
            context: None,
            name: Some("Test Subscription".to_string()),
            description: None,
            entities: vec![EntitySelector {
                id: None,
                id_pattern: None,
                entity_type: Some("Vehicle".to_string()),
            }],
            watched_attributes: None,
            notification_trigger: None,
            q: None,
            geo_q: None,
            csf: None,
            notification: NotificationParams {
                attributes: None,
                format: None,
                endpoint: NotificationEndpoint {
                    uri: "http://example.org/notify".to_string(),
                    accept: None,
                    receiver_info: None,
                    notifier_info: None,
                },
                show_changes: None,
                sys_attrs: None,
                last_notification: None,
                last_success: None,
                last_failure: None,
                times_sent: None,
                times_failed: None,
            },
            expires_at: None,
            throttling: None,
            temporal_q: None,
            scope_q: None,
            lang: None,
            status: Some(SubscriptionStatus::Active),
            is_active: Some(true),
            created_at: None,
            modified_at: None,
        }
    }

    #[test]
    fn test_validate_subscription_valid() {
        let subscription = create_test_subscription();
        assert!(validate_subscription(&subscription).is_ok());
    }

    #[test]
    fn test_validate_subscription_no_entities() {
        let mut subscription = create_test_subscription();
        subscription.entities.clear();
        assert!(validate_subscription(&subscription).is_err());
    }

    #[test]
    fn test_validate_subscription_empty_endpoint() {
        let mut subscription = create_test_subscription();
        subscription.notification.endpoint.uri = String::new();
        assert!(validate_subscription(&subscription).is_err());
    }

    #[tokio::test]
    async fn test_in_memory_store() {
        let store = InMemorySubscriptionStore::new();
        let subscription = create_test_subscription();

        // Store
        store.store_subscription(&subscription).await.unwrap();

        // Get
        let retrieved = store
            .get_subscription("urn:ngsi-ld:Subscription:test001")
            .await
            .unwrap();
        assert!(retrieved.is_some());

        // List
        let list = store.list_subscriptions(None, None).await.unwrap();
        assert_eq!(list.len(), 1);

        // Count
        let count = store.count_subscriptions().await.unwrap();
        assert_eq!(count, 1);

        // Delete
        store
            .delete_subscription("urn:ngsi-ld:Subscription:test001")
            .await
            .unwrap();
        let retrieved = store
            .get_subscription("urn:ngsi-ld:Subscription:test001")
            .await
            .unwrap();
        assert!(retrieved.is_none());
    }
}
