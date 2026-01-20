//! Usage Control
//!
//! Runtime enforcement of ODRL usage policies with event tracking.

use super::OdrlAction;
use crate::ids::types::{IdsResult, IdsUri};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Usage event for audit and compliance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageEvent {
    /// Event unique ID
    pub event_id: String,

    /// Timestamp
    pub timestamp: DateTime<Utc>,

    /// Contract ID
    pub contract_id: IdsUri,

    /// Resource accessed
    pub resource_id: IdsUri,

    /// Action performed
    pub action: OdrlAction,

    /// Party performing action
    pub party_id: IdsUri,

    /// Result (permitted or denied)
    pub result: UsageResult,

    /// Event metadata
    pub metadata: HashMap<String, String>,
}

/// Usage result
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UsageResult {
    /// Action was permitted
    Permitted,

    /// Action was denied
    Denied,

    /// Action was completed successfully
    Completed,

    /// Action failed
    Failed,
}

/// Usage statistics per contract
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageStatistics {
    /// Contract ID
    pub contract_id: IdsUri,

    /// Total usage count
    pub total_count: u64,

    /// Usage by action type
    pub by_action: HashMap<String, u64>,

    /// First usage timestamp
    pub first_used_at: Option<DateTime<Utc>>,

    /// Last usage timestamp
    pub last_used_at: Option<DateTime<Utc>>,

    /// Permitted count
    pub permitted_count: u64,

    /// Denied count
    pub denied_count: u64,
}

impl Default for UsageStatistics {
    fn default() -> Self {
        Self {
            contract_id: IdsUri::new("urn:ids:usage:unknown").unwrap(),
            total_count: 0,
            by_action: HashMap::new(),
            first_used_at: None,
            last_used_at: None,
            permitted_count: 0,
            denied_count: 0,
        }
    }
}

/// Usage Controller for runtime enforcement
pub struct UsageController {
    /// Usage event log
    event_log: Arc<RwLock<Vec<UsageEvent>>>,

    /// Usage statistics per contract
    statistics: Arc<RwLock<HashMap<String, UsageStatistics>>>,

    /// Maximum events to keep in memory
    max_events: usize,
}

impl UsageController {
    /// Create a new usage controller
    pub fn new() -> Self {
        Self {
            event_log: Arc::new(RwLock::new(Vec::new())),
            statistics: Arc::new(RwLock::new(HashMap::new())),
            max_events: 10000,
        }
    }

    /// Create with custom max events
    pub fn with_max_events(max_events: usize) -> Self {
        Self {
            event_log: Arc::new(RwLock::new(Vec::new())),
            statistics: Arc::new(RwLock::new(HashMap::new())),
            max_events,
        }
    }

    /// Record a usage event
    pub async fn record_event(&self, event: UsageEvent) -> IdsResult<()> {
        let contract_key = event.contract_id.as_str().to_string();

        // Update statistics
        {
            let mut stats = self.statistics.write().await;
            let stat = stats
                .entry(contract_key.clone())
                .or_insert_with(|| UsageStatistics {
                    contract_id: event.contract_id.clone(),
                    ..Default::default()
                });

            stat.total_count += 1;

            let action_key = format!("{:?}", event.action);
            *stat.by_action.entry(action_key).or_insert(0) += 1;

            if stat.first_used_at.is_none() {
                stat.first_used_at = Some(event.timestamp);
            }
            stat.last_used_at = Some(event.timestamp);

            match event.result {
                UsageResult::Permitted | UsageResult::Completed => {
                    stat.permitted_count += 1;
                }
                UsageResult::Denied | UsageResult::Failed => {
                    stat.denied_count += 1;
                }
            }
        }

        // Add to event log
        {
            let mut log = self.event_log.write().await;
            log.push(event);

            // Trim old events if exceeding max
            let current_len = log.len();
            if current_len > self.max_events {
                log.drain(0..(current_len - self.max_events));
            }
        }

        Ok(())
    }

    /// Get usage statistics for a contract
    pub async fn get_statistics(&self, contract_id: &IdsUri) -> Option<UsageStatistics> {
        let stats = self.statistics.read().await;
        stats.get(contract_id.as_str()).cloned()
    }

    /// Get recent usage events
    pub async fn get_recent_events(&self, limit: usize) -> Vec<UsageEvent> {
        let log = self.event_log.read().await;
        log.iter().rev().take(limit).cloned().collect()
    }

    /// Check if usage count exceeds limit
    pub async fn check_usage_limit(&self, contract_id: &IdsUri, limit: u64) -> bool {
        if let Some(stats) = self.get_statistics(contract_id).await {
            stats.total_count <= limit
        } else {
            true
        }
    }

    /// Clear usage statistics
    pub async fn clear_statistics(&self) {
        self.statistics.write().await.clear();
    }

    /// Clear event log
    pub async fn clear_events(&self) {
        self.event_log.write().await.clear();
    }
}

impl Default for UsageController {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_event() -> UsageEvent {
        UsageEvent {
            event_id: uuid::Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            contract_id: IdsUri::new("https://example.org/contract/1").unwrap(),
            resource_id: IdsUri::new("https://example.org/data/1").unwrap(),
            action: OdrlAction::Read,
            party_id: IdsUri::new("https://example.org/party/1").unwrap(),
            result: UsageResult::Permitted,
            metadata: HashMap::new(),
        }
    }

    #[tokio::test]
    async fn test_record_usage_event() {
        let controller = UsageController::new();
        let event = create_test_event();

        controller.record_event(event.clone()).await.unwrap();

        let stats = controller.get_statistics(&event.contract_id).await;
        assert!(stats.is_some());

        let stats = stats.unwrap();
        assert_eq!(stats.total_count, 1);
        assert_eq!(stats.permitted_count, 1);
    }

    #[tokio::test]
    async fn test_usage_limit() {
        let controller = UsageController::new();
        let contract_id = IdsUri::new("https://example.org/contract/limit-test").unwrap();

        for _ in 0..5 {
            let mut event = create_test_event();
            event.contract_id = contract_id.clone();
            controller.record_event(event).await.unwrap();
        }

        assert!(controller.check_usage_limit(&contract_id, 10).await);
        assert!(!controller.check_usage_limit(&contract_id, 3).await);
    }

    #[tokio::test]
    async fn test_recent_events() {
        let controller = UsageController::new();

        for _ in 0..10 {
            controller.record_event(create_test_event()).await.unwrap();
        }

        let recent = controller.get_recent_events(5).await;
        assert_eq!(recent.len(), 5);
    }
}
