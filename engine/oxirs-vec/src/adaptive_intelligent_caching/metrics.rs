//! Performance metrics for the adaptive intelligent caching system

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};

use super::types::{HistoricalMetric, TierMetrics};

/// Comprehensive cache performance metrics
#[derive(Debug, Default)]
pub struct CachePerformanceMetrics {
    /// Hit/miss statistics
    pub hit_count: AtomicU64,
    pub miss_count: AtomicU64,
    pub total_requests: AtomicU64,

    /// Latency statistics
    pub avg_hit_latency_ns: AtomicU64,
    pub avg_miss_latency_ns: AtomicU64,
    pub p99_latency_ns: AtomicU64,

    /// Throughput metrics
    pub requests_per_second: AtomicU64,
    pub bytes_per_second: AtomicU64,

    /// Cache efficiency
    pub cache_efficiency_score: f64,
    pub memory_utilization: f64,
    pub fragmentation_ratio: f64,

    /// Detailed statistics by tier
    pub tier_metrics: HashMap<u32, TierMetrics>,

    /// Time-series data for trend analysis
    pub historical_metrics: VecDeque<HistoricalMetric>,
}

impl Clone for CachePerformanceMetrics {
    fn clone(&self) -> Self {
        Self {
            hit_count: AtomicU64::new(self.hit_count.load(Ordering::SeqCst)),
            miss_count: AtomicU64::new(self.miss_count.load(Ordering::SeqCst)),
            total_requests: AtomicU64::new(self.total_requests.load(Ordering::SeqCst)),
            avg_hit_latency_ns: AtomicU64::new(self.avg_hit_latency_ns.load(Ordering::SeqCst)),
            avg_miss_latency_ns: AtomicU64::new(self.avg_miss_latency_ns.load(Ordering::SeqCst)),
            p99_latency_ns: AtomicU64::new(self.p99_latency_ns.load(Ordering::SeqCst)),
            requests_per_second: AtomicU64::new(self.requests_per_second.load(Ordering::SeqCst)),
            bytes_per_second: AtomicU64::new(self.bytes_per_second.load(Ordering::SeqCst)),
            cache_efficiency_score: self.cache_efficiency_score,
            memory_utilization: self.memory_utilization,
            fragmentation_ratio: self.fragmentation_ratio,
            tier_metrics: self.tier_metrics.clone(),
            historical_metrics: self.historical_metrics.clone(),
        }
    }
}

impl Serialize for CachePerformanceMetrics {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("CachePerformanceMetrics", 11)?;
        state.serialize_field("hit_count", &self.hit_count.load(Ordering::SeqCst))?;
        state.serialize_field("miss_count", &self.miss_count.load(Ordering::SeqCst))?;
        state.serialize_field(
            "total_requests",
            &self.total_requests.load(Ordering::SeqCst),
        )?;
        state.serialize_field(
            "avg_hit_latency_ns",
            &self.avg_hit_latency_ns.load(Ordering::SeqCst),
        )?;
        state.serialize_field(
            "avg_miss_latency_ns",
            &self.avg_miss_latency_ns.load(Ordering::SeqCst),
        )?;
        state.serialize_field(
            "p99_latency_ns",
            &self.p99_latency_ns.load(Ordering::SeqCst),
        )?;
        state.serialize_field(
            "requests_per_second",
            &self.requests_per_second.load(Ordering::SeqCst),
        )?;
        state.serialize_field(
            "bytes_per_second",
            &self.bytes_per_second.load(Ordering::SeqCst),
        )?;
        state.serialize_field("cache_efficiency_score", &self.cache_efficiency_score)?;
        state.serialize_field("memory_utilization", &self.memory_utilization)?;
        state.serialize_field("fragmentation_ratio", &self.fragmentation_ratio)?;
        state.serialize_field("tier_metrics", &self.tier_metrics)?;
        state.serialize_field("historical_metrics", &self.historical_metrics)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for CachePerformanceMetrics {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::{self, MapAccess, Visitor};
        use std::fmt;

        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "snake_case")]
        enum Field {
            HitCount,
            MissCount,
            TotalRequests,
            AvgHitLatencyNs,
            AvgMissLatencyNs,
            P99LatencyNs,
            RequestsPerSecond,
            BytesPerSecond,
            CacheEfficiencyScore,
            MemoryUtilization,
            FragmentationRatio,
            TierMetrics,
            HistoricalMetrics,
        }

        struct CachePerformanceMetricsVisitor;

        impl<'de> Visitor<'de> for CachePerformanceMetricsVisitor {
            type Value = CachePerformanceMetrics;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct CachePerformanceMetrics")
            }

            fn visit_map<V>(self, mut map: V) -> Result<CachePerformanceMetrics, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut hit_count = None;
                let mut miss_count = None;
                let mut total_requests = None;
                let mut avg_hit_latency_ns = None;
                let mut avg_miss_latency_ns = None;
                let mut p99_latency_ns = None;
                let mut requests_per_second = None;
                let mut bytes_per_second = None;
                let mut cache_efficiency_score = None;
                let mut memory_utilization = None;
                let mut fragmentation_ratio = None;
                let mut tier_metrics = None;
                let mut historical_metrics = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        Field::HitCount => {
                            if hit_count.is_some() {
                                return Err(de::Error::duplicate_field("hit_count"));
                            }
                            hit_count = Some(map.next_value::<u64>()?);
                        }
                        Field::MissCount => {
                            if miss_count.is_some() {
                                return Err(de::Error::duplicate_field("miss_count"));
                            }
                            miss_count = Some(map.next_value::<u64>()?);
                        }
                        Field::TotalRequests => {
                            if total_requests.is_some() {
                                return Err(de::Error::duplicate_field("total_requests"));
                            }
                            total_requests = Some(map.next_value::<u64>()?);
                        }
                        Field::AvgHitLatencyNs => {
                            if avg_hit_latency_ns.is_some() {
                                return Err(de::Error::duplicate_field("avg_hit_latency_ns"));
                            }
                            avg_hit_latency_ns = Some(map.next_value::<u64>()?);
                        }
                        Field::AvgMissLatencyNs => {
                            if avg_miss_latency_ns.is_some() {
                                return Err(de::Error::duplicate_field("avg_miss_latency_ns"));
                            }
                            avg_miss_latency_ns = Some(map.next_value::<u64>()?);
                        }
                        Field::P99LatencyNs => {
                            if p99_latency_ns.is_some() {
                                return Err(de::Error::duplicate_field("p99_latency_ns"));
                            }
                            p99_latency_ns = Some(map.next_value::<u64>()?);
                        }
                        Field::RequestsPerSecond => {
                            if requests_per_second.is_some() {
                                return Err(de::Error::duplicate_field("requests_per_second"));
                            }
                            requests_per_second = Some(map.next_value::<u64>()?);
                        }
                        Field::BytesPerSecond => {
                            if bytes_per_second.is_some() {
                                return Err(de::Error::duplicate_field("bytes_per_second"));
                            }
                            bytes_per_second = Some(map.next_value::<u64>()?);
                        }
                        Field::CacheEfficiencyScore => {
                            if cache_efficiency_score.is_some() {
                                return Err(de::Error::duplicate_field("cache_efficiency_score"));
                            }
                            cache_efficiency_score = Some(map.next_value()?);
                        }
                        Field::MemoryUtilization => {
                            if memory_utilization.is_some() {
                                return Err(de::Error::duplicate_field("memory_utilization"));
                            }
                            memory_utilization = Some(map.next_value()?);
                        }
                        Field::FragmentationRatio => {
                            if fragmentation_ratio.is_some() {
                                return Err(de::Error::duplicate_field("fragmentation_ratio"));
                            }
                            fragmentation_ratio = Some(map.next_value()?);
                        }
                        Field::TierMetrics => {
                            if tier_metrics.is_some() {
                                return Err(de::Error::duplicate_field("tier_metrics"));
                            }
                            tier_metrics = Some(map.next_value()?);
                        }
                        Field::HistoricalMetrics => {
                            if historical_metrics.is_some() {
                                return Err(de::Error::duplicate_field("historical_metrics"));
                            }
                            historical_metrics = Some(map.next_value()?);
                        }
                    }
                }

                Ok(CachePerformanceMetrics {
                    hit_count: AtomicU64::new(hit_count.unwrap_or(0)),
                    miss_count: AtomicU64::new(miss_count.unwrap_or(0)),
                    total_requests: AtomicU64::new(total_requests.unwrap_or(0)),
                    avg_hit_latency_ns: AtomicU64::new(avg_hit_latency_ns.unwrap_or(0)),
                    avg_miss_latency_ns: AtomicU64::new(avg_miss_latency_ns.unwrap_or(0)),
                    p99_latency_ns: AtomicU64::new(p99_latency_ns.unwrap_or(0)),
                    requests_per_second: AtomicU64::new(requests_per_second.unwrap_or(0)),
                    bytes_per_second: AtomicU64::new(bytes_per_second.unwrap_or(0)),
                    cache_efficiency_score: cache_efficiency_score.unwrap_or(0.0),
                    memory_utilization: memory_utilization.unwrap_or(0.0),
                    fragmentation_ratio: fragmentation_ratio.unwrap_or(0.0),
                    tier_metrics: tier_metrics.unwrap_or_default(),
                    historical_metrics: historical_metrics.unwrap_or_default(),
                })
            }
        }

        deserializer.deserialize_struct(
            "CachePerformanceMetrics",
            &[
                "hit_count",
                "miss_count",
                "total_requests",
                "avg_hit_latency_ns",
                "avg_miss_latency_ns",
                "p99_latency_ns",
                "requests_per_second",
                "bytes_per_second",
                "cache_efficiency_score",
                "memory_utilization",
                "fragmentation_ratio",
                "tier_metrics",
                "historical_metrics",
            ],
            CachePerformanceMetricsVisitor,
        )
    }
}
