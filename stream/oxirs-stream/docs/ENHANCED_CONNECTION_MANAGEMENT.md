# Enhanced Connection Management for OxiRS Stream

This document describes the enhanced connection management features implemented for the OxiRS streaming framework, providing production-ready resilience and performance optimization.

## Features Overview

### 1. Health Monitoring (`health_monitor.rs`)

Advanced health tracking system with:
- **Periodic Health Checks**: Configurable intervals for checking connection health
- **Statistics Tracking**: Comprehensive metrics including success rates, response times, and error counts
- **Dead Connection Detection**: Automatic identification and removal of failed connections
- **Health Status Levels**: 
  - `Healthy`: Connection operating normally
  - `Degraded`: Connection experiencing issues but still functional
  - `Unhealthy`: Connection failing health checks
  - `Dead`: Connection completely failed and should be removed
  - `Unknown`: Health status not yet determined

**Key Features**:
- Configurable failure and recovery thresholds
- Event notifications for status changes
- Historical health tracking with limited history retention
- Error type categorization and tracking

### 2. Automatic Reconnection (`reconnect.rs`)

Resilient reconnection mechanism with:
- **Exponential Backoff**: Configurable retry delays with multiplier
- **Maximum Retry Attempts**: Limit reconnection attempts to prevent infinite loops
- **Jitter**: Random delay variation to prevent thundering herd
- **Connection Failure Callbacks**: Custom handlers for connection failures
- **Multiple Retry Strategies**:
  - Exponential backoff (default)
  - Fixed delay
  - Linear backoff
  - Custom strategy support

**Configuration Options**:
```rust
ReconnectConfig {
    initial_delay: Duration::from_millis(100),
    max_delay: Duration::from_secs(60),
    multiplier: 2.0,
    max_attempts: 10,
    jitter_factor: 0.1,
    connection_timeout: Duration::from_secs(30),
}
```

### 3. Load Balancing (`connection_pool.rs`)

Multiple algorithms for distributing load across connections:
- **Round-Robin**: Simple rotation through available connections
- **Least Recently Used (LRU)**: Select connections with oldest activity
- **Random**: Random selection for even distribution
- **Least Connections**: Choose connections with lowest usage count
- **Weighted Round-Robin**: Performance-based selection using efficiency scores

**Efficiency Scoring**:
- Considers failure rate, response time, and connection weight
- Automatically adjusts weights based on connection performance
- Gradual weight recovery for previously failed connections

### 4. Failover Mechanisms (`failover.rs`)

Primary/secondary failover support with:
- **Automatic Failover**: Triggered on primary connection failures
- **Manual Failover**: API for triggering failover operations
- **Auto-Failback**: Optional automatic return to primary when recovered
- **Health-Based Decisions**: Failover decisions based on health monitoring
- **Event Notifications**: Real-time updates on failover operations

**Failover States**:
- `Primary`: Using primary connection
- `Secondary`: Failed over to secondary
- `FailingOver`: Failover in progress
- `FailingBack`: Returning to primary
- `Unavailable`: Both connections failed

## Integration with Connection Pool

The enhanced features are fully integrated into the connection pool:

```rust
// Create pool with all enhanced features
let pool = ConnectionPool::new_with_failover(
    pool_config,
    primary_factory,
    secondary_factory,
    failover_config,
).await?;

// Subscribe to health events
let mut health_events = pool.subscribe_health_events();

// Subscribe to reconnection events
let mut reconnect_events = pool.subscribe_reconnect_events();

// Register failure callback
pool.register_failure_callback(|conn_id, error, attempt| {
    Box::pin(async move {
        log::warn!("Connection {} failed: {}", conn_id, error);
    })
}).await;

// Get comprehensive statistics
let health_stats = pool.get_health_statistics().await;
let reconnect_stats = pool.get_reconnection_statistics().await;
let failover_stats = pool.get_failover_statistics().await;
```

## Performance Optimizations

1. **Adaptive Pool Sizing**: Automatically adjusts pool size based on load and response times
2. **Connection Weights**: Performance-based connection selection
3. **Efficiency Scoring**: Real-time calculation of connection efficiency
4. **Circuit Breaker Integration**: Prevents cascading failures
5. **Metrics Collection**: Comprehensive performance monitoring

## Event-Driven Architecture

All components emit events for real-time monitoring:
- Health status changes
- Reconnection attempts and outcomes
- Failover operations
- Connection lifecycle events

## Production Readiness

- **Thread-Safe**: All components use proper synchronization
- **Async/Await**: Full tokio integration for non-blocking operations
- **Error Handling**: Comprehensive error propagation and recovery
- **Logging**: Structured logging with tracing
- **Metrics**: Detailed statistics for monitoring systems
- **Configuration**: Flexible configuration for different deployment scenarios

## Example Usage

See `examples/enhanced_connection_demo.rs` for a complete demonstration of all features.

## Configuration Best Practices

1. **Health Check Intervals**: Balance between detection speed and overhead
2. **Retry Attempts**: Set reasonable limits to prevent resource exhaustion
3. **Failover Thresholds**: Avoid hair-trigger failovers while ensuring quick recovery
4. **Connection Timeouts**: Account for network latency and server load
5. **Pool Sizing**: Start conservative and enable adaptive sizing

## Monitoring Integration

The enhanced connection management provides metrics suitable for:
- Prometheus/Grafana dashboards
- CloudWatch metrics
- Custom monitoring solutions
- Alert systems

Key metrics to monitor:
- Connection health success rate
- Average response times
- Failover frequency
- Reconnection success rate
- Pool utilization
- Circuit breaker trips