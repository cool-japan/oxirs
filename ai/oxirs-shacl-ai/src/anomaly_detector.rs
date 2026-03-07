//! SHACL violation anomaly detection (rate-based).
//!
//! Tracks `ViolationEvent` instances in a sliding time window and computes
//! per-(shape, property) violation rates. When a rate deviates from the
//! stored baseline by more than `z_threshold` standard deviations the signal
//! is flagged as an anomaly.

use std::collections::{HashMap, VecDeque};

/// A single SHACL constraint violation event.
#[derive(Debug, Clone, PartialEq)]
pub struct ViolationEvent {
    pub shape_id: String,
    pub property: String,
    pub timestamp_ms: u64,
    pub severity: u8,
}

impl ViolationEvent {
    /// Create a new event.
    pub fn new(
        shape_id: impl Into<String>,
        property: impl Into<String>,
        timestamp_ms: u64,
        severity: u8,
    ) -> Self {
        Self {
            shape_id: shape_id.into(),
            property: property.into(),
            timestamp_ms,
            severity,
        }
    }
}

/// An anomaly signal for a (shape_id, property) combination.
#[derive(Debug, Clone)]
pub struct AnomalySignal {
    pub shape_id: String,
    pub property: String,
    /// Observed violations-per-second in the current window.
    pub rate: f64,
    /// Z-score relative to the stored baseline.
    pub z_score: f64,
    /// Whether the z-score exceeds the configured threshold.
    pub is_anomaly: bool,
}

/// Configuration for the anomaly detector.
#[derive(Debug, Clone)]
pub struct AnomalyConfig {
    /// Rolling window length in milliseconds.
    pub window_ms: u64,
    /// Minimum number of events before anomaly detection is applied.
    pub min_samples: usize,
    /// Z-score threshold; signals with |z| > threshold are flagged.
    pub z_threshold: f64,
}

impl Default for AnomalyConfig {
    fn default() -> Self {
        Self {
            window_ms: 60_000,
            min_samples: 5,
            z_threshold: 3.0,
        }
    }
}

/// Rate-based SHACL violation anomaly detector.
pub struct AnomalyDetector {
    config: AnomalyConfig,
    events: VecDeque<ViolationEvent>,
    /// Baseline (mean violations/s, std_dev violations/s) per "shape_id::property" key.
    baseline: HashMap<String, (f64, f64)>,
}

impl AnomalyDetector {
    /// Create a new detector with the given configuration.
    pub fn new(config: AnomalyConfig) -> Self {
        Self {
            config,
            events: VecDeque::new(),
            baseline: HashMap::new(),
        }
    }

    /// Record a new violation event.
    pub fn record(&mut self, event: ViolationEvent) {
        self.events.push_back(event);
    }

    /// Detect anomalies relative to baselines. Events older than
    /// `now_ms - window_ms` are excluded from rate computation.
    ///
    /// A signal is produced for every (shape_id, property) pair that has at
    /// least `min_samples` events in the window **or** has a stored baseline.
    pub fn detect(&mut self, now_ms: u64) -> Vec<AnomalySignal> {
        // Evict stale events first
        self.evict_old(now_ms);

        let window_s = self.config.window_ms as f64 / 1000.0;
        if window_s <= 0.0 {
            return vec![];
        }

        // Count events per key within the window
        let cutoff = now_ms.saturating_sub(self.config.window_ms);
        let mut counts: HashMap<String, (String, String, usize)> = HashMap::new();
        for ev in &self.events {
            if ev.timestamp_ms >= cutoff {
                let key = format!("{}::{}", ev.shape_id, ev.property);
                counts
                    .entry(key)
                    .and_modify(|(_, _, c)| *c += 1)
                    .or_insert_with(|| (ev.shape_id.clone(), ev.property.clone(), 1));
            }
        }

        // Also include keys that have a baseline but no current events
        for key in self.baseline.keys() {
            if !counts.contains_key(key) {
                if let Some(pos) = key.find("::") {
                    let shape_id = &key[..pos];
                    let property = &key[pos + 2..];
                    counts.insert(
                        key.clone(),
                        (shape_id.to_string(), property.to_string(), 0),
                    );
                }
            }
        }

        let mut signals = Vec::new();
        for (key, (shape_id, property, count)) in &counts {
            let rate = *count as f64 / window_s;

            if let Some(&(mean, std_dev)) = self.baseline.get(key) {
                let z = compute_z_score(rate, mean, std_dev);
                let is_anomaly = z.abs() > self.config.z_threshold;
                signals.push(AnomalySignal {
                    shape_id: shape_id.clone(),
                    property: property.clone(),
                    rate,
                    z_score: z,
                    is_anomaly,
                });
            } else if *count >= self.config.min_samples {
                // No baseline yet — report with z=0, not flagged
                signals.push(AnomalySignal {
                    shape_id: shape_id.clone(),
                    property: property.clone(),
                    rate,
                    z_score: 0.0,
                    is_anomaly: false,
                });
            }
        }
        signals
    }

    /// Remove events older than `now_ms - window_ms`. Returns count removed.
    pub fn evict_old(&mut self, now_ms: u64) -> usize {
        let cutoff = now_ms.saturating_sub(self.config.window_ms);
        let before = self.events.len();
        while let Some(front) = self.events.front() {
            if front.timestamp_ms < cutoff {
                self.events.pop_front();
            } else {
                break;
            }
        }
        before - self.events.len()
    }

    /// Store or update the baseline (mean, std_dev) for a key.
    pub fn update_baseline(&mut self, key: &str, rate: f64) {
        let entry = self.baseline.entry(key.to_string()).or_insert((rate, 0.0));
        // Exponential moving average update
        let alpha = 0.1_f64;
        let old_mean = entry.0;
        let new_mean = old_mean + alpha * (rate - old_mean);
        let new_var = (1.0 - alpha) * (entry.1 * entry.1 + alpha * (rate - old_mean).powi(2));
        *entry = (new_mean, new_var.sqrt());
    }

    /// Compute the current violations-per-second rate for the given key.
    pub fn rate_for(&self, key: &str, now_ms: u64) -> f64 {
        let cutoff = now_ms.saturating_sub(self.config.window_ms);
        let window_s = self.config.window_ms as f64 / 1000.0;
        if window_s <= 0.0 {
            return 0.0;
        }
        let count = self
            .events
            .iter()
            .filter(|e| {
                e.timestamp_ms >= cutoff
                    && format!("{}::{}", e.shape_id, e.property) == key
            })
            .count();
        count as f64 / window_s
    }

    /// Number of buffered events (not yet evicted).
    pub fn pending_count(&self) -> usize {
        self.events.len()
    }

    /// Access the stored baselines.
    pub fn baselines(&self) -> &HashMap<String, (f64, f64)> {
        &self.baseline
    }
}

/// Compute the Z-score of `value` relative to a Gaussian baseline.
///
/// Returns 0.0 when `std_dev` is effectively zero (to avoid division by zero).
pub fn compute_z_score(value: f64, mean: f64, std_dev: f64) -> f64 {
    if std_dev < 1e-12 {
        0.0
    } else {
        (value - mean) / std_dev
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> AnomalyConfig {
        AnomalyConfig {
            window_ms: 60_000,
            min_samples: 2,
            z_threshold: 2.0,
        }
    }

    fn ev(shape: &str, prop: &str, ts: u64) -> ViolationEvent {
        ViolationEvent::new(shape, prop, ts, 1)
    }

    // --- ViolationEvent ------------------------------------------------

    #[test]
    fn test_violation_event_new() {
        let e = ViolationEvent::new("sh:PersonShape", "sh:name", 1000, 2);
        assert_eq!(e.shape_id, "sh:PersonShape");
        assert_eq!(e.property, "sh:name");
        assert_eq!(e.timestamp_ms, 1000);
        assert_eq!(e.severity, 2);
    }

    #[test]
    fn test_violation_event_clone() {
        let e = ev("s", "p", 100);
        let c = e.clone();
        assert_eq!(e, c);
    }

    // --- AnomalyConfig defaults -----------------------------------------

    #[test]
    fn test_config_default() {
        let c = AnomalyConfig::default();
        assert_eq!(c.window_ms, 60_000);
        assert_eq!(c.min_samples, 5);
        assert_eq!(c.z_threshold, 3.0);
    }

    // --- AnomalyDetector basic ------------------------------------------

    #[test]
    fn test_new_empty() {
        let d = AnomalyDetector::new(cfg());
        assert_eq!(d.pending_count(), 0);
    }

    #[test]
    fn test_record_increases_count() {
        let mut d = AnomalyDetector::new(cfg());
        d.record(ev("s", "p", 1000));
        assert_eq!(d.pending_count(), 1);
        d.record(ev("s", "p", 2000));
        assert_eq!(d.pending_count(), 2);
    }

    // --- evict_old -------------------------------------------------------

    #[test]
    fn test_evict_old_removes_stale() {
        let mut d = AnomalyDetector::new(AnomalyConfig {
            window_ms: 1000,
            ..cfg()
        });
        d.record(ev("s", "p", 0));
        d.record(ev("s", "p", 500));
        d.record(ev("s", "p", 2000));
        let removed = d.evict_old(2000);
        // 0 and 500 are both < 2000 - 1000 = 1000, so removed
        assert_eq!(removed, 2);
        assert_eq!(d.pending_count(), 1);
    }

    #[test]
    fn test_evict_old_none_expired() {
        let mut d = AnomalyDetector::new(cfg());
        d.record(ev("s", "p", 59_000));
        d.record(ev("s", "p", 59_500));
        let removed = d.evict_old(60_000);
        assert_eq!(removed, 0);
        assert_eq!(d.pending_count(), 2);
    }

    #[test]
    fn test_evict_old_all() {
        let mut d = AnomalyDetector::new(AnomalyConfig {
            window_ms: 1000,
            ..cfg()
        });
        for i in 0..5u64 {
            d.record(ev("s", "p", i * 100));
        }
        let removed = d.evict_old(10_000);
        assert_eq!(removed, 5);
        assert_eq!(d.pending_count(), 0);
    }

    // --- rate_for -------------------------------------------------------

    #[test]
    fn test_rate_for_no_events() {
        let d = AnomalyDetector::new(cfg());
        let r = d.rate_for("s::p", 60_000);
        assert_eq!(r, 0.0);
    }

    #[test]
    fn test_rate_for_with_events() {
        let mut d = AnomalyDetector::new(AnomalyConfig {
            window_ms: 10_000,
            ..cfg()
        });
        // 5 events in a 10-second window = 0.5/s
        for i in 0..5u64 {
            d.record(ev("sh:S", "sh:p", 50_000 + i * 1000));
        }
        let rate = d.rate_for("sh:S::sh:p", 60_000);
        assert!((rate - 0.5).abs() < 0.01, "rate={}", rate);
    }

    // --- update_baseline ------------------------------------------------

    #[test]
    fn test_update_baseline_initial() {
        let mut d = AnomalyDetector::new(cfg());
        d.update_baseline("s::p", 1.0);
        assert!(d.baselines().contains_key("s::p"));
    }

    #[test]
    fn test_update_baseline_ema_converges() {
        let mut d = AnomalyDetector::new(cfg());
        for _ in 0..100 {
            d.update_baseline("s::p", 2.0);
        }
        let (mean, _) = d.baselines()["s::p"];
        assert!((mean - 2.0).abs() < 0.1, "mean={}", mean);
    }

    // --- detect ----------------------------------------------------------

    #[test]
    fn test_detect_empty_no_signals() {
        let mut d = AnomalyDetector::new(cfg());
        let signals = d.detect(60_000);
        assert!(signals.is_empty());
    }

    #[test]
    fn test_detect_below_min_samples_no_signal() {
        let mut d = AnomalyDetector::new(AnomalyConfig {
            min_samples: 5,
            ..cfg()
        });
        d.record(ev("s", "p", 59_000));
        let signals = d.detect(60_000);
        assert!(signals.is_empty());
    }

    #[test]
    fn test_detect_above_min_samples_signal_produced() {
        let mut d = AnomalyDetector::new(AnomalyConfig {
            min_samples: 2,
            ..cfg()
        });
        d.record(ev("s", "p", 59_000));
        d.record(ev("s", "p", 59_500));
        let signals = d.detect(60_000);
        assert_eq!(signals.len(), 1);
        assert_eq!(signals[0].shape_id, "s");
        assert_eq!(signals[0].property, "p");
    }

    #[test]
    fn test_detect_no_baseline_not_anomaly() {
        let mut d = AnomalyDetector::new(AnomalyConfig {
            min_samples: 1,
            ..cfg()
        });
        d.record(ev("s", "p", 59_000));
        let signals = d.detect(60_000);
        assert_eq!(signals.len(), 1);
        assert!(!signals[0].is_anomaly);
    }

    #[test]
    fn test_detect_anomaly_flagged() {
        let mut d = AnomalyDetector::new(AnomalyConfig {
            window_ms: 10_000,
            min_samples: 1,
            z_threshold: 2.0,
        });
        // Baseline: mean=0.1/s, std_dev=0.01
        d.baseline.insert("sh:S::sh:p".to_string(), (0.1, 0.01));
        // Inject a high rate: 50 events in 10s = 5/s, z = (5-0.1)/0.01 = 490
        for i in 0..50u64 {
            d.record(ev("sh:S", "sh:p", 50_000 + i * 200));
        }
        let signals = d.detect(60_000);
        let sig = signals.iter().find(|s| s.shape_id == "sh:S").expect("signal");
        assert!(sig.is_anomaly, "expected anomaly, rate={}", sig.rate);
    }

    #[test]
    fn test_detect_normal_not_flagged() {
        let mut d = AnomalyDetector::new(AnomalyConfig {
            window_ms: 60_000,
            min_samples: 1,
            z_threshold: 3.0,
        });
        // Baseline: mean=1.0/s, std_dev=0.5
        d.baseline.insert("s::p".to_string(), (1.0, 0.5));
        // 60 events in 60s = 1.0/s → z = 0
        for i in 0..60u64 {
            d.record(ev("s", "p", i * 1000));
        }
        let signals = d.detect(60_000);
        let sig = signals.iter().find(|s| s.shape_id == "s").expect("signal");
        assert!(!sig.is_anomaly);
    }

    // --- compute_z_score ------------------------------------------------

    #[test]
    fn test_z_score_zero_std_dev() {
        assert_eq!(compute_z_score(5.0, 3.0, 0.0), 0.0);
    }

    #[test]
    fn test_z_score_positive() {
        let z = compute_z_score(5.0, 3.0, 1.0);
        assert!((z - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_z_score_negative() {
        let z = compute_z_score(1.0, 3.0, 1.0);
        assert!((z + 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_z_score_zero_deviation() {
        let z = compute_z_score(3.0, 3.0, 1.0);
        assert!((z - 0.0).abs() < 1e-9);
    }

    // --- AnomalySignal fields -------------------------------------------

    #[test]
    fn test_signal_rate_positive() {
        let mut d = AnomalyDetector::new(AnomalyConfig {
            window_ms: 10_000,
            min_samples: 1,
            z_threshold: 2.0,
        });
        for i in 0..10u64 {
            d.record(ev("s", "p", 50_000 + i * 500));
        }
        let signals = d.detect(60_000);
        assert!(!signals.is_empty());
        assert!(signals[0].rate > 0.0);
    }

    // --- Multi-key detection -----------------------------------------------

    #[test]
    fn test_detect_multiple_keys() {
        let mut d = AnomalyDetector::new(AnomalyConfig {
            min_samples: 1,
            ..cfg()
        });
        for i in 0..3u64 {
            d.record(ev("shape_a", "prop_1", 59_000 + i * 100));
            d.record(ev("shape_b", "prop_2", 59_000 + i * 100));
        }
        let signals = d.detect(60_000);
        assert_eq!(signals.len(), 2);
    }

    #[test]
    fn test_pending_count_after_evict() {
        let mut d = AnomalyDetector::new(AnomalyConfig {
            window_ms: 1000,
            ..cfg()
        });
        d.record(ev("s", "p", 0));
        d.record(ev("s", "p", 500));
        d.record(ev("s", "p", 5000));
        d.evict_old(5000);
        assert_eq!(d.pending_count(), 1);
    }
}
