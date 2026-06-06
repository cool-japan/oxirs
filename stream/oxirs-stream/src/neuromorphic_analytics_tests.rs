//! Neuromorphic Analytics Tests
//!
//! All `#[cfg(test)]` blocks for the neuromorphic analytics modules.

#[cfg(test)]
mod tests {
    use crate::neuromorphic_analytics::{NeuromorphicAnalytics, NeuromorphicConfig};
    use crate::neuromorphic_analytics_learning::{
        apply_stdp_batch, bcm_update, compute_stdp_delta, hebbian_update,
        homeostatic_scaling_factor, oja_update, online_firing_rate,
    };
    use crate::neuromorphic_analytics_types::{SpikeEvent, STDP};
    use std::collections::HashMap;

    // ── Config defaults ────────────────────────────────────────────────────

    #[test]
    fn test_default_config() {
        let cfg = NeuromorphicConfig::default();
        assert_eq!(cfg.neuron_count, 1000);
        assert!(cfg.spike_threshold < 0.0);
        assert!(cfg.learning_rate > 0.0);
        assert!(cfg.enable_homeostasis);
    }

    // ── NeuromorphicAnalytics construction ─────────────────────────────────

    #[test]
    fn test_new_analytics_engine() {
        let cfg = NeuromorphicConfig::default();
        let _engine = NeuromorphicAnalytics::new(cfg);
    }

    // ── STDP learning rule ─────────────────────────────────────────────────

    fn default_stdp() -> STDP {
        STDP {
            potentiation_window: 20.0,
            depression_window: 20.0,
            max_weight_change: 0.1,
            learning_rate: 0.01,
            decay_constant: 10.0,
            curve_parameters: Default::default(),
        }
    }

    #[test]
    fn test_stdp_potentiation_pre_before_post() {
        let stdp = default_stdp();
        // Pre fires 5ms before post → should potentiate
        let delta = compute_stdp_delta(&stdp, 5.0);
        assert!(
            delta > 0.0,
            "Expected potentiation for +5ms delta_t, got {delta}"
        );
    }

    #[test]
    fn test_stdp_depression_post_before_pre() {
        let stdp = default_stdp();
        // Post fires 5ms before pre → should depress
        let delta = compute_stdp_delta(&stdp, -5.0);
        assert!(
            delta < 0.0,
            "Expected depression for -5ms delta_t, got {delta}"
        );
    }

    #[test]
    fn test_stdp_zero_outside_window() {
        let stdp = default_stdp();
        // 100ms is far outside the 20ms window
        let delta = compute_stdp_delta(&stdp, 100.0);
        assert_eq!(delta, 0.0);
    }

    #[test]
    fn test_stdp_batch_empty() {
        let stdp = default_stdp();
        let deltas = apply_stdp_batch(&stdp, &[], &[]);
        assert!(deltas.is_empty());
    }

    #[test]
    fn test_stdp_batch_produces_deltas() {
        let stdp = default_stdp();
        let make_spike = |neuron_id: u64, timestamp: f64| SpikeEvent {
            neuron_id,
            timestamp,
            amplitude: 80.0,
            metadata: HashMap::new(),
        };

        let pre_spikes = vec![make_spike(0, 0.0), make_spike(1, 1.0)];
        let post_spikes = vec![make_spike(2, 5.0), make_spike(3, 6.0)];

        let deltas = apply_stdp_batch(&stdp, &pre_spikes, &post_spikes);
        // 2 pre × 2 post = 4 pairs, all same-neuron check passes (IDs are different)
        assert_eq!(deltas.len(), 4);
        for d in &deltas {
            assert!(*d > 0.0, "Expected all potentiation, got {d}");
        }
    }

    // ── Hebbian learning ───────────────────────────────────────────────────

    #[test]
    fn test_hebbian_update_proportional() {
        let dw = hebbian_update(0.1, 0.8, 0.9);
        assert!((dw - 0.072).abs() < 1e-9);
    }

    #[test]
    fn test_hebbian_zero_presynaptic() {
        let dw = hebbian_update(0.1, 0.0, 0.9);
        assert_eq!(dw, 0.0);
    }

    // ── BCM rule ───────────────────────────────────────────────────────────

    #[test]
    fn test_bcm_above_threshold_potentiates() {
        // post_rate > theta → positive delta
        let dw = bcm_update(0.01, 1.0, 0.8, 0.5);
        assert!(dw > 0.0);
    }

    #[test]
    fn test_bcm_below_threshold_depresses() {
        // post_rate < theta → negative delta
        let dw = bcm_update(0.01, 1.0, 0.2, 0.5);
        assert!(dw < 0.0);
    }

    // ── Oja's rule ─────────────────────────────────────────────────────────

    #[test]
    fn test_oja_update_normalises() {
        // With weight=1.0 and equal rates the term (pre - post * w) → 0,
        // so the weight update is exactly 0.
        let dw = oja_update(0.1, 0.5, 0.5, 1.0);
        assert!(dw.abs() < 1e-9, "Expected ~0.0, got {dw}");
    }

    // ── Online firing rate ─────────────────────────────────────────────────

    #[test]
    fn test_online_firing_rate_converges() {
        let mut rate = 0.0_f64;
        for _ in 0..1000 {
            rate = online_firing_rate(rate, 1.0, 0.01);
        }
        assert!(
            (rate - 1.0).abs() < 0.01,
            "Rate should converge near 1.0, got {rate}"
        );
    }

    // ── Homeostatic scaling ────────────────────────────────────────────────

    #[test]
    fn test_homeostatic_scaling_above_target() {
        // Firing faster than target → scale factor < 1 (weight reduction)
        let factor = homeostatic_scaling_factor(0.8, 0.5, 100.0);
        assert!(
            factor < 1.0,
            "Expected scale < 1.0 for over-active neuron, got {factor}"
        );
    }

    #[test]
    fn test_homeostatic_scaling_below_target() {
        // Firing slower than target → scale factor > 1 (weight increase)
        let factor = homeostatic_scaling_factor(0.2, 0.5, 100.0);
        assert!(
            factor > 1.0,
            "Expected scale > 1.0 for under-active neuron, got {factor}"
        );
    }
}
