//! Example demonstrating enhanced connection management features
//!
//! This example shows:
//! - Health monitoring with statistics tracking
//! - Automatic reconnection with exponential backoff
//! - Load balancing across connections
//! - Failover mechanisms for high availability

use oxirs_stream::connection_pool::{
    ConnectionFactory, ConnectionPool, PoolConfig, PooledConnection,
    LoadBalancingStrategy, FailoverConfig,
};
use oxirs_stream::health_monitor::{HealthCheckConfig, HealthEvent};
use oxirs_stream::reconnect::{ReconnectConfig, ReconnectEvent};
use oxirs_stream::failover::FailoverEvent;
use anyhow::Result;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::time::{Duration, Instant};
use tokio::time::sleep;
use tracing::{info, warn, error};

/// Demo connection that can simulate failures
#[derive(Debug)]
struct DemoConnection {
    id: u32,
    created_at: Instant,
    last_activity: Instant,
    healthy: Arc<AtomicBool>,
    fail_after_count: Option<u32>,
    operation_count: Arc<AtomicU32>,
}

impl DemoConnection {
    fn new(id: u32, healthy: bool, fail_after_count: Option<u32>) -> Self {
        Self {
            id,
            created_at: Instant::now(),
            last_activity: Instant::now(),
            healthy: Arc::new(AtomicBool::new(healthy)),
            fail_after_count,
            operation_count: Arc::new(AtomicU32::new(0)),
        }
    }

    fn simulate_operation(&self) -> Result<String> {
        let count = self.operation_count.fetch_add(1, Ordering::SeqCst);
        
        // Simulate failure after certain operations
        if let Some(fail_after) = self.fail_after_count {
            if count >= fail_after {
                self.healthy.store(false, Ordering::SeqCst);
                return Err(anyhow::anyhow!("Connection {} failed after {} operations", self.id, count));
            }
        }

        // Simulate some work
        std::thread::sleep(Duration::from_millis(10));
        Ok(format!("Operation {} completed on connection {}", count, self.id))
    }
}

#[async_trait::async_trait]
impl PooledConnection for DemoConnection {
    async fn is_healthy(&self) -> bool {
        self.healthy.load(Ordering::SeqCst)
    }

    async fn close(&mut self) -> Result<()> {
        info!("Closing connection {}", self.id);
        Ok(())
    }

    fn created_at(&self) -> Instant {
        self.created_at
    }

    fn last_activity(&self) -> Instant {
        self.last_activity
    }

    fn update_activity(&mut self) {
        self.last_activity = Instant::now();
    }
}

/// Demo connection factory
struct DemoConnectionFactory {
    counter: Arc<AtomicU32>,
    failure_rate: f32,
    fail_after_ops: Option<u32>,
}

impl DemoConnectionFactory {
    fn new(failure_rate: f32, fail_after_ops: Option<u32>) -> Self {
        Self {
            counter: Arc::new(AtomicU32::new(0)),
            failure_rate,
            fail_after_ops,
        }
    }
}

#[async_trait::async_trait]
impl ConnectionFactory<DemoConnection> for DemoConnectionFactory {
    async fn create_connection(&self) -> Result<DemoConnection> {
        let id = self.counter.fetch_add(1, Ordering::SeqCst);
        
        // Simulate connection creation delay
        sleep(Duration::from_millis(50)).await;
        
        // Randomly fail based on failure rate
        if rand::random::<f32>() < self.failure_rate {
            return Err(anyhow::anyhow!("Failed to create connection {}", id));
        }

        let healthy = rand::random::<f32>() > 0.1; // 90% start healthy
        Ok(DemoConnection::new(id, healthy, self.fail_after_ops))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    info!("Starting enhanced connection pool demo");

    // Configure the connection pool with advanced features
    let pool_config = PoolConfig {
        min_connections: 3,
        max_connections: 10,
        connection_timeout: Duration::from_secs(5),
        idle_timeout: Duration::from_secs(60),
        max_lifetime: Duration::from_secs(300),
        health_check_interval: Duration::from_secs(5),
        retry_attempts: 3,
        adaptive_sizing: true,
        target_response_time_ms: 100,
        load_balancing: LoadBalancingStrategy::WeightedRoundRobin,
        enable_circuit_breaker: true,
        circuit_breaker_config: Some(Default::default()),
        enable_metrics: true,
        validation_timeout: Duration::from_secs(2),
        acquire_timeout: Duration::from_secs(10),
    };

    // Create primary and secondary factories for failover demo
    let primary_factory = Arc::new(DemoConnectionFactory::new(0.1, Some(20)));
    let secondary_factory = Arc::new(DemoConnectionFactory::new(0.05, None));

    // Create pool with failover support
    let failover_config = FailoverConfig {
        health_check_interval: Duration::from_secs(3),
        failure_threshold: 3,
        recovery_threshold: 2,
        auto_failback: true,
        ..Default::default()
    };

    let pool = Arc::new(
        ConnectionPool::new_with_failover(
            pool_config,
            primary_factory,
            secondary_factory,
            failover_config,
        )
        .await?
    );

    // Subscribe to health events
    let mut health_events = pool.subscribe_health_events();
    tokio::spawn(async move {
        while let Ok(event) = health_events.recv().await {
            match event {
                HealthEvent::ConnectionDead { connection_id, reason } => {
                    error!("Connection {} died: {}", connection_id, reason);
                }
                HealthEvent::ConnectionRecovered { connection_id } => {
                    info!("Connection {} recovered", connection_id);
                }
                HealthEvent::StatusChanged { connection_id, old_status, new_status } => {
                    warn!("Connection {} status changed from {:?} to {:?}", 
                        connection_id, old_status, new_status);
                }
                _ => {}
            }
        }
    });

    // Subscribe to reconnection events
    let mut reconnect_events = pool.subscribe_reconnect_events();
    tokio::spawn(async move {
        while let Ok(event) = reconnect_events.recv().await {
            match event {
                ReconnectEvent::AttemptSucceeded { connection_id, attempt, total_time } => {
                    info!("Reconnection succeeded for {} after {} attempts in {:?}", 
                        connection_id, attempt, total_time);
                }
                ReconnectEvent::ReconnectionExhausted { connection_id, total_attempts, .. } => {
                    error!("Reconnection exhausted for {} after {} attempts", 
                        connection_id, total_attempts);
                }
                _ => {}
            }
        }
    });

    // Register a failure callback
    pool.register_failure_callback(|conn_id, error, attempt| {
        Box::pin(async move {
            warn!("Connection {} failed (attempt {}): {}", conn_id, attempt, error);
        })
    }).await;

    // Simulate workload
    info!("Starting workload simulation...");
    
    let pool_clone = pool.clone();
    let workload_handle = tokio::spawn(async move {
        for i in 0..100 {
            match pool_clone.get_connection().await {
                Ok(mut handle) => {
                    if let Some(conn) = handle.as_mut() {
                        match conn.simulate_operation() {
                            Ok(result) => {
                                handle.record_operation(Duration::from_millis(10), true);
                                info!("Operation {}: {}", i, result);
                            }
                            Err(e) => {
                                handle.record_operation(Duration::from_millis(10), false);
                                warn!("Operation {} failed: {}", i, e);
                            }
                        }
                    }
                }
                Err(e) => {
                    error!("Failed to get connection: {}", e);
                }
            }
            
            sleep(Duration::from_millis(100)).await;
        }
    });

    // Periodically print pool status
    let pool_clone = pool.clone();
    let status_handle = tokio::spawn(async move {
        loop {
            sleep(Duration::from_secs(10)).await;
            
            let status = pool_clone.status().await;
            info!("Pool Status:");
            info!("  Total connections: {}", status.total_connections);
            info!("  Active connections: {}", status.active_connections);
            info!("  Idle connections: {}", status.idle_connections);
            info!("  Pending requests: {}", status.pending_requests);
            info!("  Utilization: {:.1}%", status.utilization_percent);
            info!("  Health: {}", if status.is_healthy { "HEALTHY" } else { "UNHEALTHY" });
            info!("  Circuit breaker: {}", if status.circuit_breaker_open { "OPEN" } else { "CLOSED" });
            
            let health_stats = pool_clone.get_health_statistics().await;
            info!("Health Statistics:");
            info!("  Healthy connections: {}/{}", 
                health_stats.healthy_connections, health_stats.total_connections);
            info!("  Success rate: {:.1}%", health_stats.success_rate);
            info!("  Avg response time: {:.2}ms", health_stats.avg_response_time_ms);
            
            if pool_clone.has_failover() {
                if let Some(failover_stats) = pool_clone.get_failover_statistics().await {
                    info!("Failover Statistics:");
                    info!("  Total failovers: {}", failover_stats.total_failovers);
                    info!("  Successful failovers: {}", failover_stats.successful_failovers);
                    info!("  Current state: {:?}", failover_stats.current_state);
                }
            }
            
            let reconnect_stats = pool_clone.get_reconnection_statistics().await;
            info!("Reconnection Statistics:");
            info!("  Total attempts: {}", reconnect_stats.total_attempts);
            info!("  Successful reconnects: {}", reconnect_stats.successful_reconnects);
            info!("  Current streak: {}", reconnect_stats.current_streak);
        }
    });

    // Simulate primary failure after 30 seconds
    tokio::spawn(async move {
        sleep(Duration::from_secs(30)).await;
        info!("Triggering manual failover...");
        if let Err(e) = pool.trigger_failover().await {
            error!("Failed to trigger failover: {}", e);
        }
    });

    // Wait for workload to complete
    workload_handle.await?;
    
    info!("Demo completed");
    Ok(())
}