//! GPU-accelerated time-series aggregation simulator (pure Rust).
//!
//! This module simulates the API surface and behaviour of GPU-accelerated
//! aggregations without requiring an actual GPU or any FFI.  All arithmetic is
//! performed on the CPU in pure Rust, but the interface mirrors what a real
//! GPU back-end would expose so that higher-level code can be written against
//! it and later swapped for a GPU implementation.

use crate::error::{TsdbError, TsdbResult};
use serde::{Deserialize, Serialize};

// ──────────────────────────────────────────────────────────────────────────────
// GpuAggOp
// ──────────────────────────────────────────────────────────────────────────────

/// Aggregation operation to perform on the GPU.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GpuAggOp {
    /// Summation of all elements.
    Sum,
    /// Arithmetic mean.
    Mean,
    /// Population variance.
    Variance,
    /// Population standard deviation.
    StdDev,
    /// Minimum value.
    Min,
    /// Maximum value.
    Max,
    /// Count of elements.
    Count,
    /// p-th percentile (0.0..=100.0).
    Percentile(f64),
    /// Weighted mean using external weight vector.
    WeightedMean(Vec<f64>),
}

// ──────────────────────────────────────────────────────────────────────────────
// GpuAggMetrics
// ──────────────────────────────────────────────────────────────────────────────

/// Runtime metrics reported by the GPU aggregator.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GpuAggMetrics {
    /// Simulated operations per second (based on element count).
    pub ops_per_second: f64,
    /// Simulated memory bandwidth in GB/s.
    pub memory_bandwidth_gbps: f64,
    /// Total number of batches processed since creation.
    pub batches_processed: u64,
}

// ──────────────────────────────────────────────────────────────────────────────
// GpuTimeSeriesAggregator
// ──────────────────────────────────────────────────────────────────────────────

/// Simulated GPU-accelerated time-series aggregator.
///
/// All operations are executed on the CPU in pure Rust; the `device_name` and
/// `batch_size` fields communicate the simulated device characteristics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuTimeSeriesAggregator {
    /// Simulated device name (e.g. `"NVIDIA A100"`, `"CPU fallback"`).
    pub device_name: String,
    /// Maximum number of elements per batch.
    pub batch_size: usize,
    /// Accumulated metrics.
    pub metrics: GpuAggMetrics,
}

impl GpuTimeSeriesAggregator {
    /// Create a new aggregator associated with `device_name`.
    pub fn new(device_name: &str, batch_size: usize) -> Self {
        Self {
            device_name: device_name.to_owned(),
            batch_size: batch_size.max(1),
            metrics: GpuAggMetrics::default(),
        }
    }

    /// Aggregate a flat slice of `f64` values using `op`.
    ///
    /// Returns an error if the slice is empty or if the operation parameters
    /// are invalid (e.g. percentile out of range, mismatched weight lengths).
    pub fn aggregate(&mut self, data: &[f64], op: GpuAggOp) -> TsdbResult<f64> {
        if data.is_empty() {
            return Err(TsdbError::Query(
                "GpuTimeSeriesAggregator::aggregate called with empty data".into(),
            ));
        }
        let result = compute_aggregate(data, &op)?;
        self.record_batch(data.len());
        Ok(result)
    }

    /// Aggregate multiple independent batches, applying the same `op` to each.
    pub fn aggregate_batch(&mut self, batches: &[Vec<f64>], op: GpuAggOp) -> TsdbResult<Vec<f64>> {
        let mut results = Vec::with_capacity(batches.len());
        for batch in batches {
            if batch.is_empty() {
                return Err(TsdbError::Query(
                    "GpuTimeSeriesAggregator::aggregate_batch encountered empty sub-batch".into(),
                ));
            }
            results.push(compute_aggregate(batch, &op)?);
            self.record_batch(batch.len());
        }
        Ok(results)
    }

    /// Roll up a `(timestamp, value)` time series into fixed time windows.
    ///
    /// Points whose timestamp falls within the same `window_secs`-aligned
    /// window are grouped and aggregated with `op`.  Returns a vector of
    /// `(window_start_timestamp, aggregated_value)` pairs sorted by timestamp.
    pub fn rollup(
        &mut self,
        series: &[(i64, f64)],
        window_secs: i64,
        op: GpuAggOp,
    ) -> TsdbResult<Vec<(i64, f64)>> {
        if series.is_empty() {
            return Ok(Vec::new());
        }
        if window_secs <= 0 {
            return Err(TsdbError::Query(format!(
                "window_secs must be positive, got {window_secs}"
            )));
        }

        // Group values by window bucket.
        let mut buckets: std::collections::BTreeMap<i64, Vec<f64>> =
            std::collections::BTreeMap::new();
        for &(ts, val) in series {
            let bucket = (ts / window_secs) * window_secs;
            buckets.entry(bucket).or_default().push(val);
        }

        let mut output = Vec::with_capacity(buckets.len());
        for (bucket_ts, values) in &buckets {
            let agg = compute_aggregate(values, &op)?;
            self.record_batch(values.len());
            output.push((*bucket_ts, agg));
        }
        Ok(output)
    }

    /// Return a reference to the accumulated performance metrics.
    pub fn metrics(&self) -> &GpuAggMetrics {
        &self.metrics
    }

    // ── private helpers ───────────────────────────────────────────────────────

    fn record_batch(&mut self, n_elements: usize) {
        self.metrics.batches_processed += 1;
        // Simulate throughput: assume 1e9 ops/sec (1 GFlops) for the "GPU".
        self.metrics.ops_per_second =
            1_000_000_000.0 / self.batch_size.max(1) as f64 * n_elements as f64;
        // Simulate 256 GB/s memory bandwidth (A100-class).
        self.metrics.memory_bandwidth_gbps = 256.0 * (n_elements as f64 / self.batch_size as f64);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// GpuDownsampler
// ──────────────────────────────────────────────────────────────────────────────

/// GPU-accelerated downsampler for time-series data.
///
/// Reduces a time series to at most `target_points` representative points by
/// dividing the series into equal-length segments and aggregating each segment
/// with the supplied [`GpuAggOp`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuDownsampler {
    /// Internal aggregator.
    pub aggregator: GpuTimeSeriesAggregator,
}

impl GpuDownsampler {
    /// Create a new downsampler backed by the default simulated GPU device.
    pub fn new() -> Self {
        Self {
            aggregator: GpuTimeSeriesAggregator::new("SimGPU", 1024),
        }
    }

    /// Downsample `series` to at most `target_points` points.
    ///
    /// If the series already has fewer points than `target_points` it is
    /// returned unchanged.  Otherwise the series is partitioned into
    /// `target_points` equal-width segments and each segment is collapsed to a
    /// single `(representative_timestamp, aggregated_value)` pair.
    pub fn downsample(
        &mut self,
        series: &[(i64, f64)],
        target_points: usize,
        op: GpuAggOp,
    ) -> TsdbResult<Vec<(i64, f64)>> {
        if target_points == 0 {
            return Err(TsdbError::Query("target_points must be at least 1".into()));
        }
        if series.len() <= target_points {
            return Ok(series.to_vec());
        }

        let chunk_size = (series.len() + target_points - 1) / target_points;
        let mut result = Vec::with_capacity(target_points);

        for chunk in series.chunks(chunk_size) {
            if chunk.is_empty() {
                continue;
            }
            // Use the first timestamp of the chunk as the representative.
            let rep_ts = chunk[0].0;
            let values: Vec<f64> = chunk.iter().map(|&(_, v)| v).collect();
            let agg = compute_aggregate(&values, &op)?;
            self.aggregator.record_batch(values.len());
            result.push((rep_ts, agg));
        }

        Ok(result)
    }
}

impl Default for GpuDownsampler {
    fn default() -> Self {
        Self::new()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Private helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Core CPU implementation of all supported aggregation operations.
fn compute_aggregate(data: &[f64], op: &GpuAggOp) -> TsdbResult<f64> {
    debug_assert!(!data.is_empty());

    match op {
        GpuAggOp::Sum => Ok(data.iter().copied().sum()),
        GpuAggOp::Mean => Ok(data.iter().copied().sum::<f64>() / data.len() as f64),
        GpuAggOp::Variance => {
            let mean = data.iter().copied().sum::<f64>() / data.len() as f64;
            let var = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
            Ok(var)
        }
        GpuAggOp::StdDev => {
            let mean = data.iter().copied().sum::<f64>() / data.len() as f64;
            let var = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
            Ok(var.sqrt())
        }
        GpuAggOp::Min => Ok(data.iter().copied().fold(f64::INFINITY, f64::min)),
        GpuAggOp::Max => Ok(data.iter().copied().fold(f64::NEG_INFINITY, f64::max)),
        GpuAggOp::Count => Ok(data.len() as f64),
        GpuAggOp::Percentile(p) => {
            if !(*p >= 0.0 && *p <= 100.0) {
                return Err(TsdbError::Query(format!(
                    "Percentile must be in [0, 100], got {p}"
                )));
            }
            let mut sorted = data.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let idx_f = (p / 100.0) * (sorted.len() - 1) as f64;
            let lo = idx_f.floor() as usize;
            let hi = idx_f.ceil() as usize;
            if lo == hi {
                Ok(sorted[lo])
            } else {
                let frac = idx_f - lo as f64;
                Ok(sorted[lo] * (1.0 - frac) + sorted[hi] * frac)
            }
        }
        GpuAggOp::WeightedMean(weights) => {
            if weights.len() != data.len() {
                return Err(TsdbError::Query(format!(
                    "WeightedMean: data length {} != weights length {}",
                    data.len(),
                    weights.len()
                )));
            }
            let total_weight: f64 = weights.iter().copied().sum();
            if total_weight == 0.0 {
                return Err(TsdbError::Query(
                    "WeightedMean: sum of weights is zero".into(),
                ));
            }
            let weighted_sum: f64 = data.iter().zip(weights.iter()).map(|(&v, &w)| v * w).sum();
            Ok(weighted_sum / total_weight)
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_aggregator() -> GpuTimeSeriesAggregator {
        GpuTimeSeriesAggregator::new("TestGPU", 64)
    }

    // ── GpuTimeSeriesAggregator::aggregate ────────────────────────────────────

    #[test]
    fn test_aggregate_sum() {
        let mut agg = make_aggregator();
        let result = agg
            .aggregate(&[1.0, 2.0, 3.0, 4.0], GpuAggOp::Sum)
            .expect("should succeed");
        assert_eq!(result, 10.0);
    }

    #[test]
    fn test_aggregate_mean() {
        let mut agg = make_aggregator();
        let result = agg
            .aggregate(&[2.0, 4.0, 6.0, 8.0], GpuAggOp::Mean)
            .expect("should succeed");
        assert_eq!(result, 5.0);
    }

    #[test]
    fn test_aggregate_variance() {
        let mut agg = make_aggregator();
        // Var([2,4,6,8]) = ((4+0+4+16)/4) = 6 ... let's compute properly.
        // Mean = 5.0; deviations: -3,-1,1,3; squares: 9,1,1,9; sum=20; /4=5.0
        let result = agg
            .aggregate(&[2.0, 4.0, 6.0, 8.0], GpuAggOp::Variance)
            .expect("should succeed");
        assert!((result - 5.0).abs() < 1e-10, "variance={result}");
    }

    #[test]
    fn test_aggregate_stddev() {
        let mut agg = make_aggregator();
        let result = agg
            .aggregate(&[2.0, 4.0, 6.0, 8.0], GpuAggOp::StdDev)
            .expect("should succeed");
        assert!((result - 5.0_f64.sqrt()).abs() < 1e-10, "stddev={result}");
    }

    #[test]
    fn test_aggregate_min_max() {
        let mut agg = make_aggregator();
        let data = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let min = agg.aggregate(&data, GpuAggOp::Min).expect("should succeed");
        let max = agg.aggregate(&data, GpuAggOp::Max).expect("should succeed");
        assert_eq!(min, 1.0);
        assert_eq!(max, 9.0);
    }

    #[test]
    fn test_aggregate_count() {
        let mut agg = make_aggregator();
        let result = agg
            .aggregate(&[10.0; 7], GpuAggOp::Count)
            .expect("should succeed");
        assert_eq!(result, 7.0);
    }

    #[test]
    fn test_aggregate_percentile_median() {
        let mut agg = make_aggregator();
        let data: Vec<f64> = (1..=9).map(|i| i as f64).collect();
        let median = agg
            .aggregate(&data, GpuAggOp::Percentile(50.0))
            .expect("should succeed");
        assert!((median - 5.0).abs() < 1e-10, "median={median}");
    }

    #[test]
    fn test_aggregate_percentile_100() {
        let mut agg = make_aggregator();
        let data: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        let p100 = agg
            .aggregate(&data, GpuAggOp::Percentile(100.0))
            .expect("should succeed");
        assert_eq!(p100, 10.0);
    }

    #[test]
    fn test_aggregate_percentile_invalid() {
        let mut agg = make_aggregator();
        let result = agg.aggregate(&[1.0, 2.0], GpuAggOp::Percentile(101.0));
        assert!(result.is_err());
    }

    #[test]
    fn test_aggregate_weighted_mean() {
        let mut agg = make_aggregator();
        // (1*1 + 2*3) / (1+3) = 7/4 = 1.75
        let result = agg
            .aggregate(&[1.0, 2.0], GpuAggOp::WeightedMean(vec![1.0, 3.0]))
            .expect("should succeed");
        assert!((result - 1.75).abs() < 1e-10, "weighted_mean={result}");
    }

    #[test]
    fn test_aggregate_weighted_mean_length_mismatch() {
        let mut agg = make_aggregator();
        let result = agg.aggregate(&[1.0, 2.0], GpuAggOp::WeightedMean(vec![1.0]));
        assert!(result.is_err());
    }

    #[test]
    fn test_aggregate_empty_error() {
        let mut agg = make_aggregator();
        let result = agg.aggregate(&[], GpuAggOp::Sum);
        assert!(result.is_err());
    }

    #[test]
    fn test_metrics_updated_after_aggregate() {
        let mut agg = make_aggregator();
        agg.aggregate(&[1.0, 2.0, 3.0], GpuAggOp::Sum)
            .expect("should succeed");
        assert_eq!(agg.metrics().batches_processed, 1);
    }

    // ── aggregate_batch ───────────────────────────────────────────────────────

    #[test]
    fn test_aggregate_batch_sums() {
        let mut agg = make_aggregator();
        let batches = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0]];
        let results = agg
            .aggregate_batch(&batches, GpuAggOp::Sum)
            .expect("should succeed");
        assert_eq!(results, vec![3.0, 7.0, 5.0]);
    }

    #[test]
    fn test_aggregate_batch_empty_sub_batch_error() {
        let mut agg = make_aggregator();
        let result = agg.aggregate_batch(&[vec![1.0], vec![]], GpuAggOp::Sum);
        assert!(result.is_err());
    }

    // ── rollup ────────────────────────────────────────────────────────────────

    #[test]
    fn test_rollup_groups_correctly() {
        let mut agg = make_aggregator();
        // 6 points: 3 in window [0,60), 3 in window [60,120).
        let series: Vec<(i64, f64)> = vec![
            (0, 1.0),
            (10, 2.0),
            (50, 3.0),
            (60, 4.0),
            (90, 5.0),
            (110, 6.0),
        ];
        let result = agg
            .rollup(&series, 60, GpuAggOp::Sum)
            .expect("should succeed");
        assert_eq!(result.len(), 2);
        // First bucket sum: 1+2+3=6, second: 4+5+6=15.
        let map: std::collections::HashMap<i64, f64> = result.into_iter().collect();
        assert_eq!(map[&0], 6.0);
        assert_eq!(map[&60], 15.0);
    }

    #[test]
    fn test_rollup_empty_series() {
        let mut agg = make_aggregator();
        let result = agg.rollup(&[], 60, GpuAggOp::Sum).expect("should succeed");
        assert!(result.is_empty());
    }

    #[test]
    fn test_rollup_invalid_window() {
        let mut agg = make_aggregator();
        let result = agg.rollup(&[(0, 1.0)], 0, GpuAggOp::Sum);
        assert!(result.is_err());
    }

    // ── GpuDownsampler ────────────────────────────────────────────────────────

    #[test]
    fn test_downsample_reduces_length() {
        let mut ds = GpuDownsampler::new();
        let series: Vec<(i64, f64)> = (0..100).map(|i| (i as i64, i as f64)).collect();
        let result = ds
            .downsample(&series, 10, GpuAggOp::Mean)
            .expect("should succeed");
        assert!(
            result.len() <= 10,
            "expected ≤10 points, got {}",
            result.len()
        );
    }

    #[test]
    fn test_downsample_passthrough_when_small() {
        let mut ds = GpuDownsampler::new();
        let series = vec![(0i64, 1.0f64), (1, 2.0), (2, 3.0)];
        let result = ds
            .downsample(&series, 10, GpuAggOp::Mean)
            .expect("should succeed");
        assert_eq!(result, series);
    }

    #[test]
    fn test_downsample_zero_target_error() {
        let mut ds = GpuDownsampler::new();
        let result = ds.downsample(&[(0, 1.0)], 0, GpuAggOp::Mean);
        assert!(result.is_err());
    }
}
