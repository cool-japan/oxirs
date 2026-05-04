//! SLA threshold parameters describing the resource budget of a single
//! [`crate::sla::SlaClass`].

use serde::{Deserialize, Serialize};

/// Resource thresholds associated with an [`crate::sla::SlaClass`].
///
/// These thresholds describe the contract that a tenant signs onto when they
/// receive a particular SLA tier.  They drive both:
///
/// * the token-bucket parameters of [`crate::sla::AdmissionController`], and
/// * the latency / concurrency budgets enforced by downstream executors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlaThresholds {
    /// p99 latency budget in milliseconds.
    pub max_latency_p99_ms: u64,
    /// Maximum simultaneous queries for a single tenant.
    pub max_concurrent_queries: usize,
    /// Maximum sustained bandwidth in MB/s.
    pub bandwidth_mb_per_sec: f64,
    /// Token-bucket refill rate (tokens/second).
    pub token_refill_rate: f64,
    /// Maximum token-bucket capacity (burst headroom).
    pub token_bucket_capacity: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thresholds_round_trip() {
        let thresholds = SlaThresholds {
            max_latency_p99_ms: 250,
            max_concurrent_queries: 8,
            bandwidth_mb_per_sec: 75.0,
            token_refill_rate: 12.0,
            token_bucket_capacity: 30.0,
        };
        let serialized = serde_json::to_string(&thresholds).expect("serialize");
        let decoded: SlaThresholds = serde_json::from_str(&serialized).expect("deserialize");
        assert_eq!(decoded.max_latency_p99_ms, thresholds.max_latency_p99_ms);
        assert_eq!(
            decoded.max_concurrent_queries,
            thresholds.max_concurrent_queries
        );
        assert!((decoded.bandwidth_mb_per_sec - thresholds.bandwidth_mb_per_sec).abs() < 1e-9);
        assert!((decoded.token_refill_rate - thresholds.token_refill_rate).abs() < 1e-9);
        assert!((decoded.token_bucket_capacity - thresholds.token_bucket_capacity).abs() < 1e-9);
    }
}
