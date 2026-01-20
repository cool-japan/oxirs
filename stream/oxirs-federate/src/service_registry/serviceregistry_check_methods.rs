//! # ServiceRegistry - check_methods Methods
//!
//! This module contains method implementations for `ServiceRegistry`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::serviceregistry_type::ServiceRegistry;
use super::types::{HealthState, HealthStatus, SparqlEndpoint};
use chrono::Utc;
use reqwest::Client;
use std::time::Instant;

impl ServiceRegistry {
    /// Check SPARQL endpoint health
    pub(super) async fn check_sparql_health(
        client: &Client,
        endpoint: &SparqlEndpoint,
    ) -> HealthStatus {
        let start = Instant::now();
        let service_id = endpoint.id.clone();
        let query = "ASK { ?s ?p ?o }";
        match client
            .post(endpoint.url.clone())
            .header("Content-Type", "application/sparql-query")
            .header("Accept", "application/sparql-results+json")
            .body(query.to_string())
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
