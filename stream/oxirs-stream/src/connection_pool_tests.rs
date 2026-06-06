//! Connection Pool — Test suite

#[cfg(test)]
mod tests {
    use crate::connection_pool::{
        CircuitBreakerConfig, ConnectionFactory, ConnectionPool, LoadBalancingStrategy, PoolConfig,
        PooledConnection,
    };
    use anyhow::Result;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::time::{Duration, Instant};
    use tokio::sync::Mutex;

    #[derive(Debug, Clone)]
    struct TestConnection {
        id: u32,
        created_at: Instant,
        last_activity: Instant,
        is_healthy: Arc<AtomicBool>,
        is_closed: bool,
    }

    impl TestConnection {
        fn new(id: u32) -> Self {
            let now = Instant::now();
            Self {
                id,
                created_at: now,
                last_activity: now,
                is_healthy: Arc::new(AtomicBool::new(true)),
                is_closed: false,
            }
        }
    }

    #[async_trait::async_trait]
    impl PooledConnection for TestConnection {
        async fn is_healthy(&self) -> bool {
            !self.is_closed && self.is_healthy.load(Ordering::Relaxed)
        }
        async fn close(&mut self) -> Result<()> {
            self.is_closed = true;
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
        fn clone_connection(&self) -> Box<dyn PooledConnection> {
            Box::new(self.clone())
        }
    }

    struct TestConnectionFactory {
        counter: Arc<Mutex<u32>>,
    }

    impl TestConnectionFactory {
        fn new() -> Self {
            Self {
                counter: Arc::new(Mutex::new(0)),
            }
        }
    }

    #[async_trait::async_trait]
    impl ConnectionFactory<TestConnection> for TestConnectionFactory {
        async fn create_connection(&self) -> Result<TestConnection> {
            let mut counter = self.counter.lock().await;
            *counter += 1;
            Ok(TestConnection::new(*counter))
        }
    }

    #[tokio::test]
    async fn test_pool_creation() {
        let config = PoolConfig {
            min_connections: 2,
            max_connections: 5,
            ..Default::default()
        };
        let factory = Arc::new(TestConnectionFactory::new());
        let pool = ConnectionPool::new(config, factory).await.unwrap();
        let status = pool.status().await;
        assert_eq!(status.idle_connections, 2);
        assert_eq!(status.active_connections, 0);
    }

    #[tokio::test]
    async fn test_connection_borrowing() {
        let config = PoolConfig {
            min_connections: 1,
            max_connections: 3,
            ..Default::default()
        };
        let factory = Arc::new(TestConnectionFactory::new());
        let pool = ConnectionPool::new(config, factory).await.unwrap();

        let mut handle = pool.get_connection().await.unwrap();
        let status = pool.status().await;
        assert_eq!(status.active_connections, 1);
        assert_eq!(status.idle_connections, 0);
        assert!(status.is_healthy);

        handle.record_operation(Duration::from_millis(50), true);
        handle.record_operation(Duration::from_millis(75), true);

        let (ops, successes, avg_time) = handle.get_operation_stats();
        assert_eq!(ops, 2);
        assert_eq!(successes, 2);
        assert!(avg_time > Duration::ZERO);

        drop(handle);
        tokio::time::sleep(Duration::from_millis(10)).await;

        let status = pool.status().await;
        assert_eq!(status.active_connections, 0);
        assert_eq!(status.idle_connections, 1);
    }

    #[tokio::test]
    async fn test_load_balancing_strategies() {
        for strategy in [
            LoadBalancingStrategy::RoundRobin,
            LoadBalancingStrategy::Random,
            LoadBalancingStrategy::LeastRecentlyUsed,
            LoadBalancingStrategy::LeastConnections,
            LoadBalancingStrategy::WeightedRoundRobin,
        ] {
            let config = PoolConfig {
                min_connections: 3,
                max_connections: 5,
                load_balancing: strategy.clone(),
                ..Default::default()
            };
            let factory = Arc::new(TestConnectionFactory::new());
            let pool = ConnectionPool::new(config, factory).await.unwrap();

            let handles: Vec<_> =
                futures_util::future::join_all((0..3).map(|_| pool.get_connection()))
                    .await
                    .into_iter()
                    .collect::<Result<Vec<_>, _>>()
                    .unwrap();

            let status = pool.status().await;
            assert_eq!(status.active_connections, 3);
            assert_eq!(status.load_balancing_strategy, strategy);

            drop(handles);
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    #[tokio::test]
    async fn test_circuit_breaker_integration() {
        let config = PoolConfig {
            min_connections: 1,
            max_connections: 2,
            enable_circuit_breaker: true,
            circuit_breaker_config: Some(CircuitBreakerConfig {
                failure_threshold: 2,
                timeout: Duration::from_millis(50),
                ..Default::default()
            }),
            ..Default::default()
        };
        let factory = Arc::new(TestConnectionFactory::new());
        let pool = ConnectionPool::new(config, factory).await.unwrap();

        let handle = pool.get_connection().await;
        assert!(handle.is_ok());
        drop(handle);

        let status = pool.status().await;
        assert!(!status.circuit_breaker_open);
    }

    #[tokio::test]
    async fn test_adaptive_sizing() {
        let config = PoolConfig {
            min_connections: 1,
            max_connections: 5,
            adaptive_sizing: true,
            target_response_time_ms: 50,
            ..Default::default()
        };
        let factory = Arc::new(TestConnectionFactory::new());
        let pool = ConnectionPool::new(config, factory).await.unwrap();

        let metrics = pool.get_detailed_metrics().await;
        assert_eq!(metrics.current_target_size, 1);
        assert!(metrics.adaptive_scaling_events == 0);

        pool.resize(3).await.unwrap();
        let metrics = pool.get_detailed_metrics().await;
        assert_eq!(metrics.current_target_size, 3);
    }

    #[tokio::test]
    async fn test_detailed_metrics() {
        let config = PoolConfig {
            min_connections: 2,
            max_connections: 4,
            enable_metrics: true,
            ..Default::default()
        };
        let factory = Arc::new(TestConnectionFactory::new());
        let pool = ConnectionPool::new(config, factory).await.unwrap();

        let handles: Vec<_> = futures_util::future::join_all((0..3).map(|_| pool.get_connection()))
            .await
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        let metrics = pool.get_detailed_metrics().await;
        assert!(metrics.total_requests >= 3);
        assert!(metrics.status.utilization_percent > 0.0);
        assert!(metrics.pool_uptime > Duration::ZERO);
        assert_eq!(metrics.status.active_connections, 3);

        drop(handles);

        pool.reset_statistics().await;
        let metrics = pool.get_detailed_metrics().await;
        assert_eq!(metrics.adaptive_scaling_events, 0);
    }
}
