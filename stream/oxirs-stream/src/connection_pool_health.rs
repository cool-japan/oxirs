//! Connection Pool — Health checking, maintenance, and adaptive sizing
//!
//! Background maintenance task, health monitoring loop, reconnection logic,
//! and adaptive sizing background task.

use crate::health_monitor::HealthStatus;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tracing::{debug, error, info, warn};

use super::connection_pool_manager::ConnectionPool;
use super::connection_pool_types::{PooledConnection, PooledConnectionWrapper};

impl<T: PooledConnection + Clone> ConnectionPool<T> {
    /// Start background maintenance task with health monitoring
    pub(super) async fn start_maintenance_task(&self) {
        let connections = self.connections.clone();
        let stats = self.stats.clone();
        let config = self.config.clone();
        let health_monitor = self.health_monitor.clone();
        let reconnect_manager = self.reconnect_manager.clone();
        let connection_factory = self.connection_factory.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(config.health_check_interval);

            loop {
                interval.tick().await;

                let mut connections_guard = connections.lock().await;
                let mut to_remove: Vec<usize> = Vec::new();
                let mut to_reconnect: Vec<(usize, String)> = Vec::new();

                for (index, wrapper) in connections_guard.iter().enumerate() {
                    let conn_id = wrapper.connection_id.clone();

                    if wrapper.is_expired(config.max_lifetime, config.idle_timeout) {
                        to_remove.push(index);
                        health_monitor.unregister_connection(&conn_id).await;
                    } else {
                        let health_status = health_monitor
                            .check_connection_health(&conn_id, &wrapper.connection)
                            .await
                            .unwrap_or(HealthStatus::Unknown);

                        match health_status {
                            HealthStatus::Dead => {
                                to_remove.push(index);
                                health_monitor.unregister_connection(&conn_id).await;
                            }
                            HealthStatus::Unhealthy => {
                                to_reconnect.push((index, conn_id.clone()));
                            }
                            _ => {}
                        }
                    }
                }

                // Remove dead connections in reverse order
                for &index in to_remove.iter().rev() {
                    if let Some(mut wrapper) = connections_guard.remove(index) {
                        if let Err(e) = wrapper.connection.close().await {
                            warn!("Failed to close connection during maintenance: {}", e);
                        }
                        stats.write().await.total_destroyed += 1;
                    }
                }

                // Attempt to reconnect unhealthy connections
                let to_reconnect_count = to_reconnect.len();
                for (index, conn_id) in &to_reconnect {
                    if *index < connections_guard.len() {
                        match reconnect_manager
                            .reconnect(conn_id.clone(), connection_factory.clone())
                            .await
                        {
                            Ok(new_conn) => {
                                let mut new_wrapper = PooledConnectionWrapper::new(new_conn);
                                new_wrapper.connection_id = conn_id.clone();

                                let mut metadata = HashMap::new();
                                metadata.insert("pool_id".to_string(), "main".to_string());
                                health_monitor
                                    .register_connection(conn_id.clone(), metadata)
                                    .await;

                                connections_guard[*index] = new_wrapper;
                                info!("Successfully reconnected connection {}", conn_id);
                            }
                            Err(e) => {
                                warn!("Failed to reconnect connection {}: {}", conn_id, e);
                                connections_guard.remove(*index);
                                stats.write().await.total_destroyed += 1;
                            }
                        }
                    }
                }

                let dead_connections = health_monitor.get_dead_connections().await;
                if !dead_connections.is_empty() {
                    warn!(
                        "Health monitor detected {} dead connections",
                        dead_connections.len()
                    );
                }

                debug!(
                    "Pool maintenance completed, removed {} connections, attempted {} reconnections",
                    to_remove.len(),
                    to_reconnect_count
                );
            }
        });
    }

    /// Start health monitoring for all connections
    pub(super) async fn start_health_monitoring(&self) {
        let connections = self.connections.lock().await;
        for wrapper in connections.iter() {
            let mut metadata = HashMap::new();
            metadata.insert("pool_id".to_string(), "main".to_string());
            metadata.insert(
                "created_at".to_string(),
                wrapper.created_at.elapsed().as_secs().to_string(),
            );
            self.health_monitor
                .register_connection(wrapper.connection_id.clone(), metadata)
                .await;
        }

        let mut health_events = self.health_monitor.subscribe();
        let stats = self.stats.clone();

        tokio::spawn(async move {
            while let Ok(event) = health_events.recv().await {
                match event {
                    crate::health_monitor::HealthEvent::ConnectionDead {
                        connection_id,
                        reason,
                    } => {
                        error!("Connection {} marked as dead: {}", connection_id, reason);
                        stats.write().await.health_check_failures += 1;
                    }
                    crate::health_monitor::HealthEvent::ConnectionRecovered { connection_id } => {
                        info!("Connection {} recovered", connection_id);
                    }
                    crate::health_monitor::HealthEvent::StatusChanged {
                        connection_id,
                        old_status,
                        new_status,
                    } => {
                        debug!(
                            "Connection {} status changed from {:?} to {:?}",
                            connection_id, old_status, new_status
                        );
                    }
                    _ => {}
                }
            }
        });
    }

    /// Start adaptive sizing background task
    pub(super) async fn start_adaptive_sizing_task(&self) {
        let pool_metrics = self.metrics.clone();
        let adaptive_controller = self.adaptive_controller.clone();
        let pool_config = self.config.clone();
        let stats = self.stats.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(30));

            loop {
                interval.tick().await;

                let metrics = pool_metrics.read().await;
                let mut controller = adaptive_controller.write().await;

                if !controller.enabled {
                    continue;
                }

                let avg_response_time = Duration::from_millis(metrics.avg_wait_time_ms as u64);
                let current_utilization =
                    if let Some((_, util)) = metrics.utilization_history.back() {
                        *util
                    } else {
                        0.0
                    };

                let should_scale_up = controller.should_scale_up(
                    metrics.current_size,
                    avg_response_time,
                    current_utilization,
                );
                let should_scale_down = controller.should_scale_down(
                    metrics.current_size,
                    avg_response_time,
                    current_utilization,
                );

                if should_scale_up && metrics.current_size < pool_config.max_connections {
                    controller.current_target_size =
                        (controller.current_target_size + 1).min(pool_config.max_connections);
                    controller.last_adjustment = Instant::now();
                    stats.write().await.adaptive_scaling_events += 1;
                    info!(
                        "Adaptive scaling: scaling UP to {}",
                        controller.current_target_size
                    );
                } else if should_scale_down && metrics.current_size > pool_config.min_connections {
                    controller.current_target_size =
                        (controller.current_target_size.saturating_sub(1))
                            .max(pool_config.min_connections);
                    controller.last_adjustment = Instant::now();
                    stats.write().await.adaptive_scaling_events += 1;
                    info!(
                        "Adaptive scaling: scaling DOWN to {}",
                        controller.current_target_size
                    );
                }
            }
        });
    }
}
