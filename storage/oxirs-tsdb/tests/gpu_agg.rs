//! Integration tests for the GPU aggregation free-function API.
//!
//! All functions are available without the `gpu` feature; they fall back to
//! CPU arithmetic transparently.  Tests are therefore always runnable.

use oxirs_tsdb::analytics::gpu_aggregations::{
    avg_column, count_column, gpu_sum, max_column, min_column, rolling_avg, rolling_sum, sum_column,
};

// ── sum_column ────────────────────────────────────────────────────────────────

#[test]
fn sum_falls_back_to_cpu_when_no_gpu_feature() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    // sum_column always succeeds — it falls back to CPU on any GPU error.
    assert_eq!(sum_column(&data), 15.0);
}

#[test]
fn sum_empty_slice_is_zero() {
    assert_eq!(sum_column(&[]), 0.0);
}

#[test]
fn sum_single_element() {
    assert!((sum_column(&[42.0]) - 42.0).abs() < 1e-10);
}

// ── gpu_sum ───────────────────────────────────────────────────────────────────

#[test]
fn gpu_sum_without_gpu_feature_returns_error() {
    let result = gpu_sum(&[1.0, 2.0]);
    #[cfg(feature = "gpu")]
    {
        // With GPU feature: may succeed or fail depending on device availability.
        // We just require the type to be correct.
        let _ = result;
    }
    #[cfg(not(feature = "gpu"))]
    {
        assert!(
            result.is_err(),
            "gpu_sum must return Err when compiled without `gpu` feature"
        );
    }
}

// ── min_column / max_column ───────────────────────────────────────────────────

#[test]
fn min_max_basic() {
    let data = vec![3.0, 1.0, 4.0, 1.0, 5.0];
    assert_eq!(min_column(&data), Some(1.0));
    assert_eq!(max_column(&data), Some(5.0));
}

#[test]
fn min_max_single_element() {
    assert_eq!(min_column(&[7.0]), Some(7.0));
    assert_eq!(max_column(&[7.0]), Some(7.0));
}

#[test]
fn min_max_empty_slice_is_none() {
    assert_eq!(min_column(&[]), None);
    assert_eq!(max_column(&[]), None);
}

#[test]
fn min_max_negative_values() {
    let data = vec![-5.0, -1.0, -3.0];
    assert_eq!(min_column(&data), Some(-5.0));
    assert_eq!(max_column(&data), Some(-1.0));
}

// ── avg_column ────────────────────────────────────────────────────────────────

#[test]
fn avg_basic() {
    let data = vec![2.0, 4.0, 6.0];
    let avg = avg_column(&data).expect("should produce a value");
    assert!((avg - 4.0).abs() < 1e-10, "avg={avg}");
}

#[test]
fn avg_empty_slice_is_none() {
    assert_eq!(avg_column(&[]), None);
}

#[test]
fn avg_single_element() {
    let avg = avg_column(&[9.0]).expect("should be Some");
    assert!((avg - 9.0).abs() < 1e-10);
}

// ── count_column ──────────────────────────────────────────────────────────────

#[test]
fn count_basic() {
    let data = vec![1.0; 7];
    assert_eq!(count_column(&data), 7);
}

#[test]
fn count_empty() {
    assert_eq!(count_column(&[]), 0);
}

// ── rolling_sum ───────────────────────────────────────────────────────────────

#[test]
fn rolling_sum_basic() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let result = rolling_sum(&data, 3);
    assert_eq!(result, vec![6.0, 9.0, 12.0]);
}

#[test]
fn rolling_sum_window_one() {
    let data = vec![1.0, 2.0, 3.0];
    // Window of 1 should equal the input slice.
    assert_eq!(rolling_sum(&data, 1), data);
}

#[test]
fn rolling_sum_window_equals_length() {
    let data = vec![1.0, 2.0, 3.0];
    let result = rolling_sum(&data, 3);
    assert_eq!(result, vec![6.0]);
}

#[test]
fn rolling_sum_window_larger_than_data_is_empty() {
    let data = vec![1.0, 2.0];
    assert!(rolling_sum(&data, 5).is_empty());
}

#[test]
fn rolling_sum_window_zero_is_empty() {
    let data = vec![1.0, 2.0, 3.0];
    assert!(rolling_sum(&data, 0).is_empty());
}

#[test]
fn rolling_sum_running_correctness() {
    // Verify element-by-element that the running sum matches a naive scan.
    let data: Vec<f64> = (1..=10).map(|i| i as f64).collect();
    let w = 4;
    let result = rolling_sum(&data, w);
    let expected: Vec<f64> = (0..=(data.len() - w))
        .map(|i| data[i..i + w].iter().sum())
        .collect();
    for (a, b) in result.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1e-10, "mismatch: {a} vs {b}");
    }
}

// ── rolling_avg ───────────────────────────────────────────────────────────────

#[test]
fn rolling_avg_basic() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let result = rolling_avg(&data, 3);
    assert_eq!(result.len(), 3);
    assert!((result[0] - 2.0).abs() < 1e-10, "result[0]={}", result[0]);
    assert!((result[1] - 3.0).abs() < 1e-10, "result[1]={}", result[1]);
    assert!((result[2] - 4.0).abs() < 1e-10, "result[2]={}", result[2]);
}

#[test]
fn rolling_avg_window_larger_than_data_is_empty() {
    assert!(rolling_avg(&[1.0, 2.0], 5).is_empty());
}

#[test]
fn rolling_avg_window_zero_is_empty() {
    assert!(rolling_avg(&[1.0], 0).is_empty());
}

// ── combined / edge-case tests ────────────────────────────────────────────────

#[test]
fn all_aggregations_on_empty_slice() {
    assert_eq!(sum_column(&[]), 0.0);
    assert_eq!(min_column(&[]), None);
    assert_eq!(max_column(&[]), None);
    assert_eq!(avg_column(&[]), None);
    assert_eq!(count_column(&[]), 0);
    assert!(rolling_sum(&[], 1).is_empty());
    assert!(rolling_avg(&[], 1).is_empty());
}

#[test]
fn sum_and_avg_consistent() {
    let data: Vec<f64> = (1..=100).map(|i| i as f64).collect();
    let sum = sum_column(&data);
    let count = count_column(&data);
    let avg = avg_column(&data).expect("non-empty");
    assert!((avg - sum / count as f64).abs() < 1e-8);
}
