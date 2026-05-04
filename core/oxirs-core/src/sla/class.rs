//! SLA class definitions for per-tenant resource allocation.
//!
//! Defines four service tiers (Bronze → Platinum) with associated latency,
//! concurrency, and token-bucket parameters.  Higher tiers receive more
//! tokens, tighter latency budgets, and higher dispatch priority.

use serde::{Deserialize, Serialize};

use super::thresholds::SlaThresholds;

// ─────────────────────────────────────────────────────────────────────────────
// SlaClass
// ─────────────────────────────────────────────────────────────────────────────

/// Ordered SLA service tier.
///
/// `PartialOrd` / `Ord` are derived so that `Bronze < Silver < Gold < Platinum`
/// — useful for comparisons and min/max in priority logic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum SlaClass {
    /// Lowest tier: shared resources, best-effort latency.
    Bronze,
    /// Standard tier: guaranteed baseline throughput.
    Silver,
    /// Premium tier: low latency, high concurrency.
    Gold,
    /// Highest tier: near-real-time, maximum concurrency.
    Platinum,
}

impl SlaClass {
    /// Return the resource thresholds for this tier.
    pub fn thresholds(&self) -> SlaThresholds {
        match self {
            SlaClass::Bronze => SlaThresholds {
                max_latency_p99_ms: 5_000,
                max_concurrent_queries: 2,
                bandwidth_mb_per_sec: 10.0,
                token_refill_rate: 1.0,
                token_bucket_capacity: 5.0,
            },
            SlaClass::Silver => SlaThresholds {
                max_latency_p99_ms: 2_000,
                max_concurrent_queries: 5,
                bandwidth_mb_per_sec: 50.0,
                token_refill_rate: 5.0,
                token_bucket_capacity: 20.0,
            },
            SlaClass::Gold => SlaThresholds {
                max_latency_p99_ms: 500,
                max_concurrent_queries: 20,
                bandwidth_mb_per_sec: 200.0,
                token_refill_rate: 20.0,
                token_bucket_capacity: 50.0,
            },
            SlaClass::Platinum => SlaThresholds {
                max_latency_p99_ms: 100,
                max_concurrent_queries: 100,
                bandwidth_mb_per_sec: 1_000.0,
                token_refill_rate: 100.0,
                token_bucket_capacity: 200.0,
            },
        }
    }

    /// Dispatch priority used by [`crate::sla::priority_dispatcher::PriorityDispatcher`].
    ///
    /// Higher value ⇒ dequeued first by the max-heap.
    pub fn dispatch_priority(&self) -> u8 {
        match self {
            SlaClass::Bronze => 1,
            SlaClass::Silver => 2,
            SlaClass::Gold => 3,
            SlaClass::Platinum => 4,
        }
    }

    /// Human-readable name for the tier.
    pub fn name(&self) -> &'static str {
        match self {
            SlaClass::Bronze => "bronze",
            SlaClass::Silver => "silver",
            SlaClass::Gold => "gold",
            SlaClass::Platinum => "platinum",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thresholds_monotonically_increasing() {
        let bronze = SlaClass::Bronze.thresholds();
        let silver = SlaClass::Silver.thresholds();
        let gold = SlaClass::Gold.thresholds();
        let platinum = SlaClass::Platinum.thresholds();

        // Latency budget tightens as tier increases
        assert!(bronze.max_latency_p99_ms > silver.max_latency_p99_ms);
        assert!(silver.max_latency_p99_ms > gold.max_latency_p99_ms);
        assert!(gold.max_latency_p99_ms > platinum.max_latency_p99_ms);

        // Concurrency, bandwidth, and token capacity all grow
        assert!(bronze.max_concurrent_queries < silver.max_concurrent_queries);
        assert!(silver.max_concurrent_queries < gold.max_concurrent_queries);
        assert!(gold.max_concurrent_queries < platinum.max_concurrent_queries);

        assert!(bronze.token_refill_rate < silver.token_refill_rate);
        assert!(silver.token_refill_rate < gold.token_refill_rate);
        assert!(gold.token_refill_rate < platinum.token_refill_rate);

        assert!(bronze.token_bucket_capacity < silver.token_bucket_capacity);
        assert!(silver.token_bucket_capacity < gold.token_bucket_capacity);
        assert!(gold.token_bucket_capacity < platinum.token_bucket_capacity);
    }

    #[test]
    fn test_dispatch_priority_ordered() {
        assert!(SlaClass::Bronze.dispatch_priority() < SlaClass::Silver.dispatch_priority());
        assert!(SlaClass::Silver.dispatch_priority() < SlaClass::Gold.dispatch_priority());
        assert!(SlaClass::Gold.dispatch_priority() < SlaClass::Platinum.dispatch_priority());
    }

    #[test]
    fn test_ord_derives_correctly() {
        assert!(SlaClass::Bronze < SlaClass::Silver);
        assert!(SlaClass::Silver < SlaClass::Gold);
        assert!(SlaClass::Gold < SlaClass::Platinum);
        assert!(SlaClass::Platinum > SlaClass::Bronze);
    }

    #[test]
    fn test_names() {
        assert_eq!(SlaClass::Bronze.name(), "bronze");
        assert_eq!(SlaClass::Silver.name(), "silver");
        assert_eq!(SlaClass::Gold.name(), "gold");
        assert_eq!(SlaClass::Platinum.name(), "platinum");
    }

    #[test]
    fn test_roundtrip_serialization() {
        let original = SlaClass::Gold;
        let json = serde_json::to_string(&original).expect("serialize");
        let decoded: SlaClass = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(original, decoded);
    }
}
