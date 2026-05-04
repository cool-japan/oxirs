//! W3-S11 — extended ML model support: regression + classification.
//!
//! Trains the new online regression and classification models on synthetic
//! streams and asserts they reach a useful prediction quality, exercising the
//! `StreamRegressor` / `StreamClassifier` traits.

#![allow(clippy::uninlined_format_args)]

use scirs2_core::ndarray_ext::Array1;

use oxirs_stream::ml::{
    GbtConfig, KnnConfig, LinearConfig, LogisticConfig, OnlineLinearRegressor,
    OnlineLogisticClassifier, StreamClassifier, StreamRegressor, StreamingGradientBoostedRegressor,
    StreamingKnnClassifier,
};

#[test]
fn online_linear_regressor_recovers_truth_to_within_tolerance() {
    let cfg = LinearConfig {
        n_features: 4,
        learning_rate: 0.05,
        l2: 0.0,
        standardise_inputs: true,
        min_samples: 10,
    };
    let model = OnlineLinearRegressor::new(cfg).expect("ok");

    // Truth: y = 1.5 x0 - 2.5 x1 + 0.7 x2 + 4.0 x3 + 3.0
    fn truth(x: &Array1<f64>) -> f64 {
        1.5 * x[0] - 2.5 * x[1] + 0.7 * x[2] + 4.0 * x[3] + 3.0
    }

    for i in 0..3000 {
        let x = Array1::from_vec(vec![
            ((i % 17) as f64) * 0.3 - 2.5,
            ((i % 13) as f64) * 0.2 - 1.5,
            ((i % 11) as f64) * 0.4 - 2.0,
            ((i % 7) as f64) * 0.5 - 1.5,
        ]);
        let y = truth(&x);
        model.observe(&x, y).expect("observe");
    }

    // Predict on a fresh probe; loss should be small.
    let probe = Array1::from_vec(vec![1.0, -1.0, 0.5, 2.0]);
    let pred = model.predict(&probe).expect("ready");
    let want = truth(&probe);
    assert!(
        (pred - want).abs() < 1.5,
        "linear regressor missed by too much: pred={pred}, want={want}"
    );
}

#[test]
fn streaming_gbt_regressor_tracks_nonlinear_target() {
    let cfg = GbtConfig {
        n_features: 2,
        max_trees: 32,
        learning_rate: 0.1,
        fit_buffer_size: 16,
        min_samples: 32,
    };
    let model = StreamingGradientBoostedRegressor::new(cfg).expect("ok");

    // Truth: piecewise — class A in upper-right quadrant, class B otherwise.
    fn truth(x: &Array1<f64>) -> f64 {
        if x[0] > 0.0 && x[1] > 0.0 {
            5.0
        } else {
            -2.0
        }
    }

    for i in 0..1500 {
        let x0 = ((i % 19) as f64) * 0.3 - 2.5;
        let x1 = ((i % 13) as f64) * 0.4 - 2.5;
        let x = Array1::from_vec(vec![x0, x1]);
        model.observe(&x, truth(&x)).expect("observe");
    }
    // Probes in each quadrant.
    let pos = Array1::from_vec(vec![1.5, 1.5]);
    let neg = Array1::from_vec(vec![-1.5, -1.5]);
    let pred_pos = model.predict(&pos).expect("ready");
    let pred_neg = model.predict(&neg).expect("ready");
    assert!(
        pred_pos > 1.0,
        "GBT failed to learn upper-quadrant class: {pred_pos}"
    );
    assert!(
        pred_neg < 1.0,
        "GBT failed to separate lower-quadrant class: {pred_neg}"
    );
}

#[test]
fn online_logistic_classifier_separates_two_clusters() {
    let cfg = LogisticConfig {
        n_features: 2,
        n_classes: 2,
        learning_rate: 0.05,
        l2: 0.0,
        min_samples: 10,
    };
    let model = OnlineLogisticClassifier::new(cfg).expect("ok");
    fn label(x: &Array1<f64>) -> usize {
        // Two clusters separated by x0 = 0.
        if x[0] >= 0.0 {
            1
        } else {
            0
        }
    }

    for i in 0..3000 {
        let x0 = ((i % 23) as f64) * 0.3 - 3.5;
        let x1 = ((i % 17) as f64) * 0.2 - 1.5;
        let x = Array1::from_vec(vec![x0, x1]);
        model.observe(&x, label(&x)).expect("observe");
    }

    let mut correct = 0;
    let mut total = 0;
    for i in 0..400 {
        let x0 = ((i * 5) as f64).sin() * 2.0;
        let x1 = ((i * 7) as f64).cos();
        let x = Array1::from_vec(vec![x0, x1]);
        let pred = model.predict(&x).expect("ready");
        if pred.label == label(&x) {
            correct += 1;
        }
        total += 1;
    }
    let accuracy = correct as f64 / total as f64;
    assert!(accuracy > 0.9, "logistic accuracy too low: {accuracy}");
}

#[test]
fn streaming_knn_classifier_three_clusters() {
    let cfg = KnnConfig {
        n_features: 2,
        n_classes: 3,
        k: 5,
        window_size: 600,
        distance_weighted: false,
        min_samples: 30,
    };
    let model = StreamingKnnClassifier::new(cfg).expect("ok");

    // Three radial clusters.
    let centres = [(0.0, 0.0), (10.0, 0.0), (5.0, 8.0)];
    for i in 0..600 {
        let cluster = i % 3;
        let c = centres[cluster];
        let dx = ((i / 3) as f64).sin() * 0.3;
        let dy = ((i / 3) as f64).cos() * 0.3;
        let x = Array1::from_vec(vec![c.0 + dx, c.1 + dy]);
        model.observe(&x, cluster).expect("observe");
    }
    // Sample directly at each centre.
    for (cluster, c) in centres.iter().enumerate() {
        let probe = Array1::from_vec(vec![c.0, c.1]);
        let pred = model.predict(&probe).expect("ready");
        assert_eq!(
            pred.label, cluster,
            "centre of cluster {cluster} misclassified"
        );
    }
}

#[test]
fn knn_distance_weighted_resolves_ties_in_favour_of_closer_neighbour() {
    let cfg = KnnConfig {
        n_features: 1,
        n_classes: 2,
        k: 3,
        window_size: 5,
        distance_weighted: true,
        min_samples: 3,
    };
    let model = StreamingKnnClassifier::new(cfg).expect("ok");
    model
        .observe(&Array1::from_vec(vec![0.0]), 0)
        .expect("observe");
    model
        .observe(&Array1::from_vec(vec![5.0]), 1)
        .expect("observe");
    model
        .observe(&Array1::from_vec(vec![6.0]), 1)
        .expect("observe");

    let probe = Array1::from_vec(vec![0.05]);
    let pred = model.predict(&probe).expect("ready");
    assert_eq!(
        pred.label, 0,
        "distance-weighted vote should prefer near class"
    );
    let total: f64 = pred.scores.iter().sum();
    assert!((total - 1.0).abs() < 1e-6 || total == 0.0);
}

#[test]
fn ml_traits_compose_through_dynamic_dispatch() {
    let regs: Vec<Box<dyn StreamRegressor>> = vec![
        Box::new(
            OnlineLinearRegressor::new(LinearConfig {
                n_features: 1,
                ..Default::default()
            })
            .expect("ok"),
        ),
        Box::new(
            StreamingGradientBoostedRegressor::new(GbtConfig {
                n_features: 1,
                ..Default::default()
            })
            .expect("ok"),
        ),
    ];

    for r in &regs {
        for i in 0..200 {
            let x = Array1::from_vec(vec![i as f64 * 0.1]);
            r.observe(&x, x[0] * 2.0 + 1.0).expect("observe");
        }
        let probe = Array1::from_vec(vec![5.0]);
        let pred = r.predict(&probe).expect("ready");
        assert!(pred.is_finite());
    }

    let clfs: Vec<Box<dyn StreamClassifier>> = vec![
        Box::new(OnlineLogisticClassifier::new(LogisticConfig::default()).expect("ok")),
        Box::new(StreamingKnnClassifier::new(KnnConfig::default()).expect("ok")),
    ];
    for c in &clfs {
        for i in 0..200 {
            let x = Array1::from_vec(vec![
                ((i % 13) as f64) * 0.1,
                ((i % 7) as f64) * 0.2,
                ((i % 5) as f64) * 0.3,
                ((i % 3) as f64) * 0.5,
            ]);
            let label = if x[0] > 0.5 { 1 } else { 0 };
            c.observe(&x, label).expect("observe");
        }
    }
}
