//! # ServiceRegistry - start_group Methods
//!
//! This module contains method implementations for `ServiceRegistry`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::serviceregistry_type::ServiceRegistry;
use anyhow::Result;
use chrono::Utc;
use std::sync::Arc;
use tracing::info;

impl ServiceRegistry {
    /// Start the service registry with health monitoring
    pub async fn start(&mut self) -> Result<()> {
        info!("Starting service registry");
        if self.health_monitor_handle.is_none() {
            let handle = self.start_health_monitoring().await;
            self.health_monitor_handle = Some(handle);
        }
        self.start_capability_monitoring().await;
        Ok(())
    }
    /// Start health monitoring
    async fn start_health_monitoring(&self) -> tokio::task::JoinHandle<()> {
        let sparql_endpoints = Arc::clone(&self.sparql_endpoints);
        let graphql_services = Arc::clone(&self.graphql_services);
        let health_status = Arc::clone(&self.health_status);
        let http_client = self.http_client.clone();
        let interval = self.config.health_check_interval;
        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);
            loop {
                interval_timer.tick().await;
                for entry in sparql_endpoints.iter() {
                    let endpoint = entry.value();
                    let health = Self::check_sparql_health(&http_client, endpoint).await;
                    health_status.insert(endpoint.id.clone(), health);
                }
                for entry in graphql_services.iter() {
                    let service = entry.value();
                    let health = Self::check_graphql_health(&http_client, service).await;
                    health_status.insert(service.id.clone(), health);
                }
            }
        })
    }
    /// Start capability monitoring
    async fn start_capability_monitoring(&self) {
        let capabilities_cache = Arc::clone(&self.capabilities_cache);
        let _sparql_endpoints = Arc::clone(&self.sparql_endpoints);
        let _graphql_services = Arc::clone(&self.graphql_services);
        let interval = self.config.capability_refresh_interval;
        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);
            loop {
                interval_timer.tick().await;
                let now = Utc::now();
                let mut cache = capabilities_cache.write();
                cache.retain(|_, entry| entry.expires_at > now);
            }
        });
    }
}
