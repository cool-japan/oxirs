# OxiRS Cluster Performance Tuning Guide

**Version:** 0.1.0-beta.2
**Last Updated:** November 20, 2025

## Overview

This guide provides comprehensive recommendations for optimizing OxiRS Cluster performance across different deployment scenarios, from small development clusters to large-scale production environments.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Cloud Storage Optimization](#cloud-storage-optimization)
3. [Disaster Recovery Tuning](#disaster-recovery-tuning)
4. [Elastic Scaling Configuration](#elastic-scaling-configuration)
5. [ML Cost Optimization](#ml-cost-optimization)
6. [Monitoring & Metrics](#monitoring--metrics)
7. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Development Environment (< 10 nodes)

```toml
[cluster]
min_nodes = 3
max_nodes = 5
target_cpu_utilization = 0.70

[storage]
default_tier = "hot"
encryption_enabled = false  # Disable for development
compression_level = 3

[replication]
replication_factor = 1  # Single copy for development
sync_writes = false
```

### Production Environment (10-100 nodes)

```toml
[cluster]
min_nodes = 10
max_nodes = 100
target_cpu_utilization = 0.75

[storage]
default_tier = "hot"
encryption_enabled = true
compression_level = 6

[replication]
replication_factor = 3
sync_writes = true
rto_seconds = 300  # 5 minutes
rpo_seconds = 60   # 1 minute
```

### Large-Scale Production (100+ nodes)

```toml
[cluster]
min_nodes = 100
max_nodes = 1000
target_cpu_utilization = 0.80

[storage]
default_tier = "warm"  # Cost optimization
encryption_enabled = true
compression_level = 9

[replication]
replication_factor = 5
sync_writes = false  # Async for performance
batch_size = 1000
```

---

## Cloud Storage Optimization

### S3 Backend Tuning

**Upload Performance:**
- Use `StorageTier::Hot` for frequently accessed data
- Enable multipart upload for files > 5MB
- Set appropriate upload buffer size (default: 8MB)

```rust
let config = CloudStorageConfig {
    provider: CloudProvider::AWS,
    region: "us-east-1".to_string(),
    bucket: "oxirs-production".to_string(),
    default_tier: StorageTier::Hot,
    encryption_enabled: true,
    versioning_enabled: false,  // Disable if not needed
    lifecycle_rules: vec![
        LifecycleRule {
            id: "transition_to_cold".to_string(),
            transition_days: 90,
            target_tier: StorageTier::Cold,
            enabled: true,
        },
    ],
};
```

**Download Performance:**
- Enable connection pooling
- Use parallel downloads for large objects
- Consider CloudFront/CDN for frequently accessed data

**Cost Optimization:**
- Use `StorageTier::Warm` for infrequently accessed data (70% cost savings)
- Use `StorageTier::Cold` for archival (90% cost savings)
- Implement lifecycle policies to automatically transition data

### GCS Backend Tuning

**Regional Optimization:**
```rust
// Choose region closest to compute resources
let config = CloudStorageConfig {
    provider: CloudProvider::GCP,
    region: "us-central1".to_string(),  // Same region as compute
    // ... other config
};
```

**Performance Tiers:**
- Standard: Default for hot data
- Nearline: Data accessed < 1/month (40% cheaper)
- Coldline: Data accessed < 1/quarter (70% cheaper)
- Archive: Long-term archival (90% cheaper)

### Azure Blob Storage Tuning

**Access Tiers:**
- Hot: Optimized for frequent access
- Cool: Infrequent access (50% storage cost savings)
- Archive: Rare access (95% storage cost savings)

**Performance:**
- Enable Azure CDN for global distribution
- Use premium block blobs for low-latency requirements

---

## Disaster Recovery Tuning

### RTO/RPO Configuration

**Aggressive (Mission-Critical):**
```rust
let config = DisasterRecoveryConfig {
    rto_seconds: 60,        // 1 minute recovery
    rpo_seconds: 10,        // 10 seconds data loss max
    auto_failover_enabled: true,
    health_check_interval_secs: 10,
    failover_threshold: 2,  // Fail after 2 checks
    continuous_replication: true,
    replication_batch_size: 100,
    ..Default::default()
};
```

**Balanced (Standard Production):**
```rust
let config = DisasterRecoveryConfig {
    rto_seconds: 300,       // 5 minutes recovery
    rpo_seconds: 60,        // 1 minute data loss max
    auto_failover_enabled: true,
    health_check_interval_secs: 30,
    failover_threshold: 3,
    continuous_replication: true,
    replication_batch_size: 1000,
    ..Default::default()
};
```

**Cost-Optimized (Non-Critical):**
```rust
let config = DisasterRecoveryConfig {
    rto_seconds: 1800,      // 30 minutes recovery
    rpo_seconds: 300,       // 5 minutes data loss max
    auto_failover_enabled: false,  // Manual failover
    health_check_interval_secs: 120,
    continuous_replication: false,
    replication_batch_size: 10000,
    ..Default::default()
};
```

### Multi-Cloud Strategy

**Primary-Secondary Pattern:**
```
AWS (Primary) -> GCP (Secondary) -> Azure (Tertiary)
```

**Active-Active Pattern (High Availability):**
```
AWS (Active) <-> GCP (Active)
- Both serving traffic
- Bi-directional replication
- Geographic load balancing
```

---

## Elastic Scaling Configuration

### Auto-Scaling Thresholds

**Conservative (Cost-Optimized):**
```rust
let config = ElasticScalingConfig {
    min_nodes: 5,
    max_nodes: 20,
    target_cpu_utilization: 0.80,       // Higher threshold
    target_memory_utilization: 0.85,
    scale_up_threshold: 0.85,
    scale_down_threshold: 0.40,         // More aggressive scale-down
    cooldown_seconds: 600,              // 10 minutes
    use_spot_instances: true,
    max_spot_ratio: 0.70,               // 70% spot instances
    ..Default::default()
};
```

**Aggressive (Performance-Optimized):**
```rust
let config = ElasticScalingConfig {
    min_nodes: 10,
    max_nodes: 200,
    target_cpu_utilization: 0.60,       // Lower threshold - more headroom
    target_memory_utilization: 0.65,
    scale_up_threshold: 0.70,
    scale_down_threshold: 0.30,
    cooldown_seconds: 180,              // 3 minutes - faster response
    use_spot_instances: true,
    max_spot_ratio: 0.30,               // Lower ratio for stability
    ..Default::default()
};
```

### Spot Instance Optimization

**Maximum Cost Savings:**
- Use 70-80% spot instances
- Multiple instance types for availability
- Implement graceful shutdown handlers

**Balanced Performance/Cost:**
- Use 50% spot instances
- Critical nodes on on-demand
- Maintain hot standbys

**High Availability:**
- Use 20-30% spot instances
- All critical paths on on-demand
- Spot for batch/background jobs only

---

## ML Cost Optimization

### Training Data Collection

```rust
let optimizer = MLCostOptimizer::new();

// Collect comprehensive training data
for metric in historical_metrics {
    optimizer.add_training_data(CostTrainingData {
        instance_type: metric.instance_type.clone(),
        cpu_utilization: metric.cpu,
        memory_utilization: metric.memory,
        queries_per_second: metric.qps,
        actual_cost: metric.hourly_cost,
        is_spot: metric.is_spot_instance,
        timestamp: metric.timestamp,
    }).await;
}

// Train model (requires >= 100 samples)
optimizer.train_model().await?;
```

### Cost Predictions

```rust
// Get cost prediction with confidence
let prediction = optimizer.predict_cost(&current_metrics, &config).await;

if prediction.confidence > 0.8 {
    println!("High confidence prediction:");
    println!("  Hourly cost: ${:.2}", prediction.predicted_hourly_cost);
    println!("  Monthly savings: ${:.2}", prediction.estimated_monthly_savings);
    println!("  Recommended instance: {}", prediction.recommended_instance_type);
    println!("  Recommended spot ratio: {:.1}%", prediction.recommended_spot_ratio * 100.0);
}
```

### Cost Optimization Recommendations

```rust
let recommendations = optimizer.get_recommendations(&status, &cost_optimization).await;

for rec in recommendations {
    if rec.ml_based && rec.confidence > 0.75 {
        println!("[{}] {}", rec.impact, rec.action);
        println!("  Potential savings: ${:.2}/month", rec.predicted_savings);
        println!("  {}", rec.description);
    }
}
```

---

## Monitoring & Metrics

### Key Performance Indicators

**Cluster Health:**
- Node availability: > 99.9%
- Replication lag: < 1 second
- Query latency p99: < 100ms
- Error rate: < 0.1%

**Storage Metrics:**
- Upload throughput: Monitor for degradation
- Download latency: p95 < 50ms
- Compression ratio: Track over time
- Storage costs: $ per GB per month

**Disaster Recovery:**
- Failover time: Should match RTO
- Data loss: Should match RPO
- Health check success rate: > 99%
- Replication throughput: Bytes/sec

### Prometheus Integration

```rust
let profiler = CloudOperationProfiler::new();

// Operations are automatically tracked
profiler.start_operation("s3_upload");
// ... perform upload ...
profiler.stop_operation("s3_upload", bytes_uploaded, success);

// Export for Prometheus
let metrics = profiler.export_prometheus();
```

### Custom Metrics

```rust
let summary = backend.get_metrics_summary();

// Track custom metrics
metrics_registry.record("oxirs_uploads_total", summary.total_uploads);
metrics_registry.record("oxirs_upload_bytes", summary.total_upload_bytes);
metrics_registry.record("oxirs_avg_latency_ms", summary.avg_latency_ms);
metrics_registry.record("oxirs_compression_ratio", summary.compression_ratio);
```

---

## Troubleshooting

### High Latency

**Symptoms:** Query latency > 100ms p99

**Diagnosis:**
1. Check replication lag: `cluster.get_replication_lag()`
2. Monitor CPU/memory: Should be < 80%
3. Check network latency between regions
4. Review slow query logs

**Solutions:**
- Scale up if CPU/memory high
- Add read replicas for read-heavy workloads
- Use closer cloud regions
- Enable query result caching

### High Costs

**Symptoms:** Monthly cloud bill exceeding budget

**Diagnosis:**
1. Review storage tier distribution
2. Check spot instance ratio
3. Analyze data transfer costs
4. Review unused resources

**Solutions:**
```rust
// Get cost optimization recommendations
let recommendations = manager.get_cost_optimization().await;
for rec in recommendations.recommendations {
    println!("{}: {}", rec.priority, rec.recommendation);
}
```

- Increase spot instance usage to 50-70%
- Implement lifecycle policies for cold storage
- Use regional endpoints to reduce data transfer
- Right-size instance types based on actual usage

### Replication Lag

**Symptoms:** RPO violations, stale reads

**Diagnosis:**
1. Check network bandwidth between regions
2. Monitor replication queue depth
3. Review batch sizes
4. Check for network partitions

**Solutions:**
- Increase replication batch size
- Add more network bandwidth
- Use async replication for non-critical data
- Enable compression for replication traffic

### Failover Issues

**Symptoms:** Failover takes longer than RTO

**Diagnosis:**
1. Check health check frequency
2. Review failover threshold
3. Monitor provider availability
4. Check DNS propagation time

**Solutions:**
- Decrease health check interval
- Lower failover threshold
- Pre-warm secondary providers
- Use global load balancer for faster DNS updates

---

## Performance Benchmarks

### Cloud Storage Operations

Typical performance (measured with criterion):

```
S3 Upload (1MB):      ~50ms  (20 MB/s)
S3 Download (1MB):    ~30ms  (33 MB/s)
GCS Upload (1MB):     ~55ms  (18 MB/s)
Azure Upload (1MB):   ~60ms  (17 MB/s)

GPU Compression (1MB): ~10ms  (100 MB/s)
CPU Compression (1MB): ~40ms  (25 MB/s)
```

### Scaling Operations

```
Disaster Recovery Failover: <30 seconds
Elastic Scaling Decision:   <1 second
ML Cost Prediction:         <500ms
Health Check (3 providers): <2 seconds
```

---

## Best Practices

1. **Start with conservative settings** and tune based on metrics
2. **Enable monitoring** from day one
3. **Test disaster recovery** regularly (quarterly minimum)
4. **Review costs weekly** and adjust spot ratios
5. **Train ML models** with >= 1000 samples for best predictions
6. **Use property-based testing** for configuration validation
7. **Monitor compression ratios** and adjust levels based on data types
8. **Implement gradual rollouts** for configuration changes
9. **Keep RTO/RPO** aligned with business requirements
10. **Document all tuning changes** and their impact

---

## Additional Resources

- [OxiRS Cluster README](../README.md)
- [Cloud Integration API](../src/cloud_integration.rs)
- [ML Optimization Guide](../src/ml_optimization.rs)
- [Property-Based Tests](../tests/property_based_tests.rs)

For questions or issues, please file an issue at:
https://github.com/cool-japan/oxirs/issues
