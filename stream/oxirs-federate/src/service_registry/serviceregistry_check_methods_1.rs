//! # ServiceRegistry - check_methods Methods
//!
//! This module contains method implementations for `ServiceRegistry`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::serviceregistry_type::ServiceRegistry;
use super::types::{GraphQLService, HealthState, HealthStatus};
use chrono::Utc;
use reqwest::Client;
use std::time::Instant;

impl ServiceRegistry {
    /// Check GraphQL service health
    pub(super) async fn check_graphql_health(
        client: &Client,
        service: &GraphQLService,
    ) -> HealthStatus {
        let start = Instant::now();
        let service_id = service.id.clone();
        let query = r#"{ __schema { queryType { name } } }"#;
        match client
            .post(service.url.clone())
            .header("Content-Type", "application/json")
            .json(&serde_json::json!({ "query" : query }))
            .send()
            .await
        {
            Ok(response) => {
                let response_time = start.elapsed().as_millis() as u64;
                if response.status().is_success() {
                    HealthStatus {
                        service_id,
                        status: HealthState::Healthy,
                        last_check: Utc::now(),
                        consecutive_failures: 0,
                        last_error: None,
                        response_time_ms: Some(response_time),
                    }
                } else {
                    HealthStatus {
                        service_id,
                        status: HealthState::Degraded,
                        last_check: Utc::now(),
                        consecutive_failures: 1,
                        last_error: Some(format!("HTTP {}", response.status())),
                        response_time_ms: Some(response_time),
                    }
                }
            }
            Err(e) => HealthStatus {
                service_id,
                status: HealthState::Unhealthy,
                last_check: Utc::now(),
                consecutive_failures: 1,
                last_error: Some(e.to_string()),
                response_time_ms: None,
            },
        }
    }
}
