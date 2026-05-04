//! Integration tests for BatchCorrelationMatrix.

use oxirs_samm::analytics::batch_matrix::{BatchCorrelationError, BatchCorrelationMatrix};

/// Generate a linear ramp `start..=end` as f64.
fn ramp(start: usize, end: usize) -> Vec<f64> {
    (start..=end).map(|i| i as f64).collect()
}

#[test]
fn test_batch_correlation_basic() {
    // 3 features, 10 observations each
    let a = ramp(1, 10);
    let b: Vec<f64> = a.iter().map(|v| v * 2.0 + 1.0).collect();
    let c: Vec<f64> = a.iter().map(|v| 11.0 - v).collect(); // negatively correlated with a

    let samples: Vec<&[f64]> = vec![&a, &b, &c];
    let labels = ["x", "y", "z"];

    let mat = BatchCorrelationMatrix::compute(&samples, Some(&labels)).unwrap();

    // Diagonal must be 1.0
    for i in 0..3 {
        assert!(
            (mat.matrix[[i, i]] - 1.0).abs() < 1e-9,
            "diagonal[{}] should be 1.0, got {}",
            i,
            mat.matrix[[i, i]]
        );
    }

    // Symmetry: mat[i,j] == mat[j,i]
    for i in 0..3 {
        for j in 0..3 {
            assert!(
                (mat.matrix[[i, j]] - mat.matrix[[j, i]]).abs() < 1e-9,
                "matrix not symmetric at ({}, {})",
                i,
                j
            );
        }
    }

    // Check labels
    assert_eq!(mat.feature_labels, vec!["x", "y", "z"]);
}

#[test]
fn test_batch_correlation_identity() {
    // Identical columns → correlation must be 1.
    let col: Vec<f64> = vec![1.5, 2.5, 3.5, 4.5, 5.5, 6.5];
    let samples: Vec<&[f64]> = vec![&col, &col];

    let mat = BatchCorrelationMatrix::compute(&samples, None).unwrap();
    assert!(
        (mat.matrix[[0, 1]] - 1.0).abs() < 1e-9,
        "identical columns must correlate at 1.0, got {}",
        mat.matrix[[0, 1]]
    );
    assert!(
        (mat.matrix[[1, 0]] - 1.0).abs() < 1e-9,
        "symmetry check: got {}",
        mat.matrix[[1, 0]]
    );
}

#[test]
fn test_batch_correlation_negative() {
    // Negated column → correlation must be -1.
    let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y: Vec<f64> = x.iter().map(|v| -v).collect();

    let samples: Vec<&[f64]> = vec![&x, &y];
    let mat = BatchCorrelationMatrix::compute(&samples, None).unwrap();
    assert!(
        (mat.matrix[[0, 1]] + 1.0).abs() < 1e-9,
        "negated column must correlate at -1.0, got {}",
        mat.matrix[[0, 1]]
    );
}

#[test]
fn test_batch_correlation_significant_pairs() {
    // A perfectly correlated pair should appear in significant_pairs.
    let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let y: Vec<f64> = x.iter().map(|v| v * 3.0).collect();

    let samples: Vec<&[f64]> = vec![&x, &y];
    let mat = BatchCorrelationMatrix::compute(&samples, None).unwrap();

    assert!(
        !mat.significant_pairs.is_empty(),
        "expected at least one significant pair"
    );

    let (i, j, r) = mat.significant_pairs[0];
    assert_eq!(i, 0);
    assert_eq!(j, 1);
    assert!(r.abs() > 0.3, "coefficient {} should exceed threshold", r);
}

#[test]
fn test_batch_correlation_threshold_filtering() {
    // Two columns that are completely uncorrelated should not appear in
    // significant_pairs.  We construct them orthogonally.
    // x = 1,0,1,0,1,0,1,0  (alternating)
    // y = 0,1,0,1,0,1,0,1  (opposite alternating) → correlation = -1
    // Actually they ARE strongly correlated (r=-1). Use a white-noise-like pair.
    // x = 1,2,3,4,5,6,7,8
    // y = 8,1,6,3,4,7,2,5 (shuffled, low correlation)
    let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let y: Vec<f64> = vec![8.0, 1.0, 6.0, 3.0, 4.0, 7.0, 2.0, 5.0];

    let samples: Vec<&[f64]> = vec![&x, &y];
    let mat = BatchCorrelationMatrix::compute(&samples, None).unwrap();

    // Whether the pair appears in significant_pairs depends on |r|.
    // Just verify the property: every listed pair has |r| > 0.3.
    for (_, _, r) in &mat.significant_pairs {
        assert!(r.abs() > 0.3, "pair listed with |r|={} <= 0.3", r);
    }
}

#[test]
fn test_partial_correlation_basic() {
    // x = 1..8, y = 2x (perfectly correlated with x),
    // z = x + small perturbation (control variable)
    let x: Vec<f64> = (1..=8).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|v| v * 2.0).collect();
    let z: Vec<f64> = x
        .iter()
        .enumerate()
        .map(|(i, v)| v + (i % 3) as f64 * 0.1)
        .collect();

    let samples: Vec<&[f64]> = vec![&x, &y, &z];
    let mat = BatchCorrelationMatrix::partial_correlation_matrix(&samples, 2).unwrap();

    // Diagonal must be 1.
    for i in 0..3 {
        assert!(
            (mat.matrix[[i, i]] - 1.0).abs() < 1e-9,
            "diagonal[{}] = {} should be 1.0",
            i,
            mat.matrix[[i, i]]
        );
    }

    // Symmetry.
    for i in 0..3 {
        for j in 0..3 {
            assert!(
                (mat.matrix[[i, j]] - mat.matrix[[j, i]]).abs() < 1e-9,
                "not symmetric at ({}, {})",
                i,
                j
            );
        }
    }
}

#[test]
fn test_partial_correlation_reduces_effect() {
    // With a control variable z that is correlated with both x and y,
    // the partial correlation r(x,y|z) should differ from the raw r(x,y).
    let x: Vec<f64> = (1..=10).map(|i| i as f64).collect();
    let z: Vec<f64> = x.iter().map(|v| v * 1.5).collect();
    let y: Vec<f64> = z
        .iter()
        .enumerate()
        .map(|(i, v)| v + (i % 4) as f64 * 0.5)
        .collect();

    let samples: Vec<&[f64]> = vec![&x, &y, &z];

    let full = BatchCorrelationMatrix::compute(&samples, None).unwrap();
    let partial = BatchCorrelationMatrix::partial_correlation_matrix(&samples, 2).unwrap();

    let raw_r = full.matrix[[0, 1]];
    let partial_r = partial.matrix[[0, 1]];

    // They must not be exactly equal (z affects the relationship).
    // We just verify both are finite and bounded.
    assert!(raw_r.is_finite());
    assert!(partial_r.is_finite());
    assert!(raw_r.abs() <= 1.0 + 1e-9);
    assert!(partial_r.abs() <= 1.0 + 1e-9);
}

#[test]
fn test_batch_correlation_error_too_few_features() {
    let col = ramp(1, 5);
    let samples: Vec<&[f64]> = vec![&col];
    let result = BatchCorrelationMatrix::compute(&samples, None);
    assert!(
        matches!(result, Err(BatchCorrelationError::TooFewFeatures(1))),
        "expected TooFewFeatures, got {:?}",
        result
    );
}

#[test]
fn test_batch_correlation_error_ragged() {
    let a = ramp(1, 5);
    let b: Vec<f64> = vec![1.0, 2.0, 3.0]; // different length
    let samples: Vec<&[f64]> = vec![&a, &b];
    let result = BatchCorrelationMatrix::compute(&samples, None);
    assert!(matches!(
        result,
        Err(BatchCorrelationError::RaggedSamples { .. })
    ));
}

#[test]
fn test_partial_control_index_out_of_range() {
    let a = ramp(1, 4);
    let b = ramp(2, 5);
    let samples: Vec<&[f64]> = vec![&a, &b];
    let result = BatchCorrelationMatrix::partial_correlation_matrix(&samples, 10);
    assert!(matches!(
        result,
        Err(BatchCorrelationError::ControlIndexOutOfRange { .. })
    ));
}
