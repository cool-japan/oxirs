// Delta encoding for vector updates (v1.1.0 round 11)
//
// Implements incremental embedding updates via delta encoding:
// delta = new - old, with optional compression and quantization.

/// A delta (difference) between two version of a vector
#[derive(Debug, Clone)]
pub struct VectorDelta {
    /// Identifier of the vector this delta belongs to
    pub id: usize,
    /// Per-component differences (`delta[i] = new[i] - old[i]`)
    pub delta: Vec<f32>,
    /// Scale factor used for quantization (max absolute value of the delta)
    pub scale: f32,
    /// Creation timestamp (milliseconds since epoch or arbitrary counter)
    pub timestamp: u64,
}

impl VectorDelta {
    /// Create a new delta directly
    pub fn new(id: usize, delta: Vec<f32>, scale: f32, timestamp: u64) -> Self {
        Self {
            id,
            delta,
            scale,
            timestamp,
        }
    }

    /// True if all components are zero
    pub fn is_zero(&self) -> bool {
        self.delta.iter().all(|&v| v == 0.0)
    }
}

/// Statistics about a sequence of deltas
#[derive(Debug, Clone)]
pub struct DeltaStats {
    pub total_deltas: usize,
    pub applied: usize,
    /// RMS magnitude across all deltas
    pub magnitude: f32,
    /// Mean absolute change per component per delta
    pub mean_change: f32,
}

/// Stateless delta encoding utilities
pub struct DeltaEncoder;

impl DeltaEncoder {
    /// Encode the difference between `old` and `new` as a `VectorDelta`.
    ///
    /// `scale` is set to the maximum absolute value in the delta (useful for quantization).
    /// If `old` and `new` have different lengths, the shorter is padded with 0.0.
    pub fn encode(old: &[f32], new: &[f32]) -> VectorDelta {
        let len = old.len().max(new.len());
        let delta: Vec<f32> = (0..len)
            .map(|i| {
                let o = old.get(i).copied().unwrap_or(0.0);
                let n = new.get(i).copied().unwrap_or(0.0);
                n - o
            })
            .collect();

        let scale = delta.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);

        VectorDelta::new(0, delta, scale, 0)
    }

    /// Apply a single delta to `base`, returning the updated vector.
    ///
    /// If `base` and `delta.delta` have different lengths, the shorter is padded.
    pub fn apply(base: &[f32], delta: &VectorDelta) -> Vec<f32> {
        let len = base.len().max(delta.delta.len());
        (0..len)
            .map(|i| {
                let b = base.get(i).copied().unwrap_or(0.0);
                let d = delta.delta.get(i).copied().unwrap_or(0.0);
                b + d
            })
            .collect()
    }

    /// Apply a sequence of deltas in order to `base`.
    pub fn apply_sequence(base: &[f32], deltas: &[VectorDelta]) -> Vec<f32> {
        deltas
            .iter()
            .fold(base.to_vec(), |acc, d| Self::apply(&acc, d))
    }

    /// Zero out delta components whose absolute value is below `threshold`.
    pub fn compress(delta: &VectorDelta, threshold: f32) -> VectorDelta {
        let compressed: Vec<f32> = delta
            .delta
            .iter()
            .map(|&v| if v.abs() < threshold { 0.0 } else { v })
            .collect();

        let new_scale = compressed.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);

        VectorDelta::new(delta.id, compressed, new_scale, delta.timestamp)
    }

    /// Merge multiple deltas into a single combined delta by summing components.
    ///
    /// Returns `None` if `deltas` is empty.
    /// The merged delta gets `id = 0`, `timestamp = 0`, and the scale of the sum.
    pub fn merge(deltas: &[VectorDelta]) -> Option<VectorDelta> {
        if deltas.is_empty() {
            return None;
        }

        let len = deltas.iter().map(|d| d.delta.len()).max().unwrap_or(0);
        let mut sum = vec![0.0f32; len];

        for d in deltas {
            for (i, &v) in d.delta.iter().enumerate() {
                sum[i] += v;
            }
        }

        let scale = sum.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
        Some(VectorDelta::new(0, sum, scale, 0))
    }

    /// Compute statistics about a sequence of deltas.
    pub fn stats(deltas: &[VectorDelta]) -> DeltaStats {
        let total_deltas = deltas.len();

        if deltas.is_empty() {
            return DeltaStats {
                total_deltas: 0,
                applied: 0,
                magnitude: 0.0,
                mean_change: 0.0,
            };
        }

        // Magnitude: RMS of all delta components across all deltas
        let all_vals: Vec<f32> = deltas
            .iter()
            .flat_map(|d| d.delta.iter().copied())
            .collect();
        let sq_sum: f32 = all_vals.iter().map(|v| v * v).sum();
        let magnitude = if all_vals.is_empty() {
            0.0
        } else {
            (sq_sum / all_vals.len() as f32).sqrt()
        };

        // Mean absolute change
        let abs_sum: f32 = all_vals.iter().map(|v| v.abs()).sum();
        let mean_change = if all_vals.is_empty() {
            0.0
        } else {
            abs_sum / all_vals.len() as f32
        };

        DeltaStats {
            total_deltas,
            applied: total_deltas,
            magnitude,
            mean_change,
        }
    }

    /// Compute the L2 magnitude of a single delta (Euclidean norm).
    pub fn magnitude(delta: &VectorDelta) -> f32 {
        let sq_sum: f32 = delta.delta.iter().map(|v| v * v).sum();
        sq_sum.sqrt()
    }

    /// Quantize the delta to int8 range [-127, 127].
    ///
    /// Uses the stored `scale` as the normalisation factor.
    /// A scale of 0.0 results in all-zero output.
    pub fn quantize_to_i8(delta: &VectorDelta) -> Vec<i8> {
        if delta.scale == 0.0 {
            return vec![0i8; delta.delta.len()];
        }
        delta
            .delta
            .iter()
            .map(|&v| {
                let quantized = (v / delta.scale * 127.0).round();
                quantized.clamp(-127.0, 127.0) as i8
            })
            .collect()
    }

    /// Reconstruct a `VectorDelta` from int8 quantized values.
    /// `scale` must be the same scale used during `quantize_to_i8`.
    pub fn from_i8(quantized: &[i8], scale: f32, id: usize, timestamp: u64) -> VectorDelta {
        let delta: Vec<f32> = quantized
            .iter()
            .map(|&v| (v as f32) / 127.0 * scale)
            .collect();
        VectorDelta::new(id, delta, scale, timestamp)
    }
}

#[cfg(test)]
mod tests {
    type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
    use super::*;

    const EPS: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    fn approx_vec(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| approx_eq(*x, *y))
    }

    // ── encode / apply round-trip ─────────────────────────────────────────

    #[test]
    fn test_encode_apply_round_trip() {
        let old = vec![1.0, 2.0, 3.0];
        let new = vec![2.0, 3.0, 4.0];
        let delta = DeltaEncoder::encode(&old, &new);
        let result = DeltaEncoder::apply(&old, &delta);
        assert!(approx_vec(&result, &new));
    }

    #[test]
    fn test_encode_zero_delta() {
        let v = vec![1.0, 2.0, 3.0];
        let delta = DeltaEncoder::encode(&v, &v);
        assert!(delta.is_zero());
        assert!(approx_eq(delta.scale, 0.0));
    }

    #[test]
    fn test_encode_delta_values() {
        let old = vec![0.0, 0.0, 0.0];
        let new = vec![1.0, -2.0, 3.0];
        let delta = DeltaEncoder::encode(&old, &new);
        assert!(approx_eq(delta.delta[0], 1.0));
        assert!(approx_eq(delta.delta[1], -2.0));
        assert!(approx_eq(delta.delta[2], 3.0));
    }

    #[test]
    fn test_encode_scale_is_max_abs() {
        let old = vec![0.0, 0.0, 0.0];
        let new = vec![1.0, 5.0, -3.0];
        let delta = DeltaEncoder::encode(&old, &new);
        assert!(approx_eq(delta.scale, 5.0));
    }

    #[test]
    fn test_encode_different_lengths_pads_shorter() {
        let old = vec![1.0, 2.0];
        let new = vec![2.0, 3.0, 4.0];
        let delta = DeltaEncoder::encode(&old, &new);
        assert_eq!(delta.delta.len(), 3);
        assert!(approx_eq(delta.delta[2], 4.0)); // 4.0 - 0.0
    }

    // ── apply ─────────────────────────────────────────────────────────────

    #[test]
    fn test_apply_zero_delta_unchanged() {
        let base = vec![1.0, 2.0, 3.0];
        let delta = VectorDelta::new(0, vec![0.0, 0.0, 0.0], 0.0, 0);
        let result = DeltaEncoder::apply(&base, &delta);
        assert!(approx_vec(&result, &base));
    }

    #[test]
    fn test_apply_positive_delta() {
        let base = vec![1.0, 2.0];
        let delta = VectorDelta::new(0, vec![0.5, -0.5], 0.5, 0);
        let result = DeltaEncoder::apply(&base, &delta);
        assert!(approx_eq(result[0], 1.5));
        assert!(approx_eq(result[1], 1.5));
    }

    // ── apply_sequence ────────────────────────────────────────────────────

    #[test]
    fn test_apply_sequence_ordered() {
        let base = vec![0.0, 0.0];
        let d1 = VectorDelta::new(0, vec![1.0, 2.0], 2.0, 1);
        let d2 = VectorDelta::new(0, vec![0.5, 0.5], 0.5, 2);
        let result = DeltaEncoder::apply_sequence(&base, &[d1, d2]);
        assert!(approx_eq(result[0], 1.5));
        assert!(approx_eq(result[1], 2.5));
    }

    #[test]
    fn test_apply_sequence_empty_deltas() {
        let base = vec![1.0, 2.0, 3.0];
        let result = DeltaEncoder::apply_sequence(&base, &[]);
        assert!(approx_vec(&result, &base));
    }

    #[test]
    fn test_apply_sequence_single_delta() {
        let base = vec![1.0];
        let d = VectorDelta::new(0, vec![1.0], 1.0, 0);
        let result = DeltaEncoder::apply_sequence(&base, &[d]);
        assert!(approx_eq(result[0], 2.0));
    }

    // ── compress ──────────────────────────────────────────────────────────

    #[test]
    fn test_compress_zeroes_below_threshold() {
        let delta = VectorDelta::new(0, vec![0.5, 0.01, -0.001, 1.0], 1.0, 0);
        let compressed = DeltaEncoder::compress(&delta, 0.05);
        assert!(approx_eq(compressed.delta[1], 0.0));
        assert!(approx_eq(compressed.delta[2], 0.0));
        assert!(approx_eq(compressed.delta[0], 0.5));
        assert!(approx_eq(compressed.delta[3], 1.0));
    }

    #[test]
    fn test_compress_threshold_zero_unchanged() {
        let delta = VectorDelta::new(0, vec![0.5, 0.01, 1.0], 1.0, 0);
        let compressed = DeltaEncoder::compress(&delta, 0.0);
        assert!(approx_vec(&compressed.delta, &delta.delta));
    }

    #[test]
    fn test_compress_all_below_threshold() {
        let delta = VectorDelta::new(0, vec![0.001, 0.002, 0.003], 0.003, 0);
        let compressed = DeltaEncoder::compress(&delta, 0.01);
        assert!(compressed.is_zero());
        assert!(approx_eq(compressed.scale, 0.0));
    }

    // ── merge ─────────────────────────────────────────────────────────────

    #[test]
    fn test_merge_empty_returns_none() {
        assert!(DeltaEncoder::merge(&[]).is_none());
    }

    #[test]
    fn test_merge_single_delta() -> Result<()> {
        let d = VectorDelta::new(0, vec![1.0, 2.0], 2.0, 0);
        let merged =
            DeltaEncoder::merge(&[d]).expect("merge of non-empty slice should return Some");
        assert!(approx_eq(merged.delta[0], 1.0));
        assert!(approx_eq(merged.delta[1], 2.0));
        Ok(())
    }

    #[test]
    fn test_merge_sums_deltas() -> Result<()> {
        let d1 = VectorDelta::new(0, vec![1.0, 2.0], 2.0, 0);
        let d2 = VectorDelta::new(0, vec![3.0, 4.0], 4.0, 0);
        let merged =
            DeltaEncoder::merge(&[d1, d2]).expect("merge of non-empty slice should return Some");
        assert!(approx_eq(merged.delta[0], 4.0));
        assert!(approx_eq(merged.delta[1], 6.0));
        Ok(())
    }

    #[test]
    fn test_merge_scale_is_max_abs_sum() -> Result<()> {
        let d1 = VectorDelta::new(0, vec![1.0], 1.0, 0);
        let d2 = VectorDelta::new(0, vec![-3.0], 3.0, 0);
        let merged =
            DeltaEncoder::merge(&[d1, d2]).expect("merge of non-empty slice should return Some");
        // sum = -2.0 → scale = 2.0
        assert!(approx_eq(merged.scale, 2.0));
        Ok(())
    }

    // ── stats ─────────────────────────────────────────────────────────────

    #[test]
    fn test_stats_empty() {
        let s = DeltaEncoder::stats(&[]);
        assert_eq!(s.total_deltas, 0);
        assert!(approx_eq(s.magnitude, 0.0));
    }

    #[test]
    fn test_stats_total_deltas() {
        let deltas: Vec<_> = (0..3)
            .map(|i| VectorDelta::new(i, vec![1.0, 1.0], 1.0, 0))
            .collect();
        let s = DeltaEncoder::stats(&deltas);
        assert_eq!(s.total_deltas, 3);
    }

    #[test]
    fn test_stats_magnitude_positive() {
        let d = VectorDelta::new(0, vec![1.0, 0.0], 1.0, 0);
        let s = DeltaEncoder::stats(&[d]);
        assert!(s.magnitude > 0.0);
    }

    #[test]
    fn test_stats_mean_change_positive() {
        let d = VectorDelta::new(0, vec![1.0, 2.0], 2.0, 0);
        let s = DeltaEncoder::stats(&[d]);
        assert!(s.mean_change > 0.0);
    }

    // ── quantize_to_i8 / from_i8 ─────────────────────────────────────────

    #[test]
    fn test_quantize_to_i8_range() {
        let delta = VectorDelta::new(0, vec![1.0, -1.0, 0.5, -0.5], 1.0, 0);
        let q = DeltaEncoder::quantize_to_i8(&delta);
        for &v in &q {
            assert!(v >= -127);
        }
    }

    #[test]
    fn test_quantize_max_value() {
        let delta = VectorDelta::new(0, vec![1.0], 1.0, 0);
        let q = DeltaEncoder::quantize_to_i8(&delta);
        assert_eq!(q[0], 127);
    }

    #[test]
    fn test_quantize_min_value() {
        let delta = VectorDelta::new(0, vec![-1.0], 1.0, 0);
        let q = DeltaEncoder::quantize_to_i8(&delta);
        assert_eq!(q[0], -127);
    }

    #[test]
    fn test_quantize_zero_scale_all_zeros() {
        let delta = VectorDelta::new(0, vec![0.0, 0.0], 0.0, 0);
        let q = DeltaEncoder::quantize_to_i8(&delta);
        assert!(q.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_from_i8_reconstruction_approx() {
        let original = vec![1.0, -1.0, 0.5];
        let old = vec![0.0, 0.0, 0.0];
        let delta = DeltaEncoder::encode(&old, &original);
        let q = DeltaEncoder::quantize_to_i8(&delta);
        let reconstructed = DeltaEncoder::from_i8(&q, delta.scale, 0, 0);
        // Reconstruction should be close (within quantization error ≈ scale/127)
        for (a, b) in original.iter().zip(reconstructed.delta.iter()) {
            assert!(
                (a - b).abs() < delta.scale / 100.0 + 1e-4,
                "Reconstruction error too large: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_from_i8_preserves_id_and_timestamp() {
        let q = vec![10i8, -20i8];
        let d = DeltaEncoder::from_i8(&q, 1.0, 42, 99999);
        assert_eq!(d.id, 42);
        assert_eq!(d.timestamp, 99999);
    }

    // ── magnitude ─────────────────────────────────────────────────────────

    #[test]
    fn test_magnitude_zero_delta() {
        let delta = VectorDelta::new(0, vec![0.0, 0.0, 0.0], 0.0, 0);
        assert!(approx_eq(DeltaEncoder::magnitude(&delta), 0.0));
    }

    #[test]
    fn test_magnitude_unit_vector() {
        let delta = VectorDelta::new(0, vec![1.0, 0.0, 0.0], 1.0, 0);
        assert!(approx_eq(DeltaEncoder::magnitude(&delta), 1.0));
    }

    #[test]
    fn test_magnitude_known_value() {
        // [3, 4] → magnitude = 5
        let delta = VectorDelta::new(0, vec![3.0, 4.0], 4.0, 0);
        assert!(approx_eq(DeltaEncoder::magnitude(&delta), 5.0));
    }

    // ── VectorDelta helpers ───────────────────────────────────────────────

    #[test]
    fn test_vector_delta_is_zero_true() {
        let d = VectorDelta::new(0, vec![0.0, 0.0], 0.0, 0);
        assert!(d.is_zero());
    }

    #[test]
    fn test_vector_delta_is_zero_false() {
        let d = VectorDelta::new(0, vec![0.0, 1.0], 1.0, 0);
        assert!(!d.is_zero());
    }

    #[test]
    fn test_encode_empty_vectors() {
        let delta = DeltaEncoder::encode(&[], &[]);
        assert!(delta.delta.is_empty());
        assert!(approx_eq(delta.scale, 0.0));
    }

    // ── additional coverage ───────────────────────────────────────────────

    #[test]
    fn test_encode_preserves_length_when_equal() {
        let old = vec![1.0, 2.0, 3.0, 4.0];
        let new = vec![5.0, 6.0, 7.0, 8.0];
        let delta = DeltaEncoder::encode(&old, &new);
        assert_eq!(delta.delta.len(), 4);
    }

    #[test]
    fn test_apply_extends_base_for_longer_delta() {
        let base = vec![1.0];
        let delta = VectorDelta::new(0, vec![0.5, 0.5, 0.5], 0.5, 0);
        let result = DeltaEncoder::apply(&base, &delta);
        assert_eq!(result.len(), 3);
        assert!(approx_eq(result[0], 1.5));
        assert!(approx_eq(result[1], 0.5)); // 0.0 + 0.5
        assert!(approx_eq(result[2], 0.5)); // 0.0 + 0.5
    }

    #[test]
    fn test_apply_sequence_three_deltas() {
        let base = vec![0.0];
        let d1 = VectorDelta::new(0, vec![1.0], 1.0, 1);
        let d2 = VectorDelta::new(0, vec![2.0], 2.0, 2);
        let d3 = VectorDelta::new(0, vec![3.0], 3.0, 3);
        let result = DeltaEncoder::apply_sequence(&base, &[d1, d2, d3]);
        assert!(approx_eq(result[0], 6.0));
    }

    #[test]
    fn test_merge_three_deltas_sums_all() -> Result<()> {
        let d1 = VectorDelta::new(1, vec![1.0, 0.0], 1.0, 0);
        let d2 = VectorDelta::new(2, vec![2.0, 0.0], 2.0, 0);
        let d3 = VectorDelta::new(3, vec![3.0, 0.0], 3.0, 0);
        let merged = DeltaEncoder::merge(&[d1, d2, d3])
            .expect("merge of non-empty slice should return Some");
        assert!(approx_eq(merged.delta[0], 6.0));
        assert_eq!(merged.id, 0); // merged id is always 0
        Ok(())
    }

    #[test]
    fn test_quantize_mid_value() {
        // 0.5 / 1.0 * 127 = 63.5 → rounds to 64
        let delta = VectorDelta::new(0, vec![0.5], 1.0, 0);
        let q = DeltaEncoder::quantize_to_i8(&delta);
        assert_eq!(q[0], 64);
    }

    #[test]
    fn test_from_i8_zero_values() {
        let q = vec![0i8, 0i8, 0i8];
        let d = DeltaEncoder::from_i8(&q, 2.0, 0, 0);
        assert!(d.delta.iter().all(|&v| approx_eq(v, 0.0)));
    }

    #[test]
    fn test_stats_applied_equals_total() {
        let deltas: Vec<_> = (0..5)
            .map(|i| VectorDelta::new(i, vec![1.0], 1.0, 0))
            .collect();
        let s = DeltaEncoder::stats(&deltas);
        assert_eq!(s.applied, s.total_deltas);
        assert_eq!(s.total_deltas, 5);
    }
}
