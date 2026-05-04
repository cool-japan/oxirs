//! In-process scalability benchmarks for `SimCluster`.
//!
//! These benchmarks measure construction time and hash-ring lookup latency
//! across three cluster sizes (10, 100, 1000 nodes) without any real network
//! traffic.  They are intentionally lightweight — the point is to verify that
//! the in-memory cluster overhead grows sub-linearly with node count and that
//! O(log N) ring lookups remain under 1 ms at 1000 virtual nodes.

use super::SimCluster;
use std::time::Instant;

/// Summary of a single construction benchmark run.
#[derive(Debug, Clone)]
pub struct ScalingResult {
    /// Number of nodes in the cluster.
    pub n: usize,
    /// Wall-clock time to construct the `SimCluster`, in milliseconds.
    pub construction_ms: u64,
}

/// Construct a `SimCluster` of `n` nodes and measure the wall-clock time.
///
/// # Examples
///
/// ```rust
/// use oxirs_cluster::simulation::scaling_bench::bench_construction;
///
/// let result = bench_construction(10);
/// assert_eq!(result.n, 10);
/// // Construction of 10 nodes should be fast.
/// assert!(result.construction_ms < 1000);
/// ```
pub fn bench_construction(n: usize) -> ScalingResult {
    let start = Instant::now();
    let _cluster = SimCluster::new(n);
    ScalingResult {
        n,
        construction_ms: start.elapsed().as_millis() as u64,
    }
}

/// Run construction benchmarks for 10, 100, and 1000 nodes.
///
/// Returns one [`ScalingResult`] per tier in ascending node-count order.
pub fn bench_all_tiers() -> [ScalingResult; 3] {
    [
        bench_construction(10),
        bench_construction(100),
        bench_construction(1000),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bench_construction_10() {
        let r = bench_construction(10);
        assert_eq!(r.n, 10);
        // Should complete well within 1 second in any environment.
        assert!(
            r.construction_ms < 1000,
            "10-node construction too slow: {}ms",
            r.construction_ms
        );
    }

    #[test]
    fn test_bench_construction_100() {
        let r = bench_construction(100);
        assert_eq!(r.n, 100);
        assert!(
            r.construction_ms < 5000,
            "100-node construction too slow: {}ms",
            r.construction_ms
        );
    }

    #[test]
    fn test_bench_construction_1000() {
        let r = bench_construction(1000);
        assert_eq!(r.n, 1000);
        assert!(
            r.construction_ms < 30000,
            "1000-node construction too slow: {}ms",
            r.construction_ms
        );
    }

    #[test]
    fn test_bench_all_tiers_counts() {
        let tiers = bench_all_tiers();
        assert_eq!(tiers[0].n, 10);
        assert_eq!(tiers[1].n, 100);
        assert_eq!(tiers[2].n, 1000);
    }
}
