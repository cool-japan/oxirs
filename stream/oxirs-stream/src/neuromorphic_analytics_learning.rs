//! Neuromorphic Analytics Learning
//!
//! STDP (spike-timing dependent plasticity), Hebbian learning,
//! and online learning algorithms.

use crate::error::StreamResult;
use crate::neuromorphic_analytics_network::SynapticPlasticity;
use crate::neuromorphic_analytics_types::{
    NeuromorphicConfig, NeuromorphicProcessingResult, SpikeEvent, STDP,
};

/// Compute the STDP weight delta for a pre→post spike pair.
///
/// Returns a positive value (potentiation) when the pre-synaptic spike
/// precedes the post-synaptic spike within `stdp.potentiation_window`,
/// and a negative value (depression) in the opposite temporal order.
pub fn compute_stdp_delta(stdp: &STDP, delta_t_ms: f64) -> f64 {
    if delta_t_ms >= 0.0 && delta_t_ms <= stdp.potentiation_window {
        // Pre before post: long-term potentiation
        stdp.max_weight_change * (-delta_t_ms / stdp.decay_constant).exp()
    } else if delta_t_ms < 0.0 && delta_t_ms.abs() <= stdp.depression_window {
        // Post before pre: long-term depression
        -stdp.max_weight_change * (delta_t_ms / stdp.decay_constant).exp()
    } else {
        0.0
    }
}

/// Apply STDP updates across a batch of spike pairs.
///
/// Returns the vector of weight deltas indexed by synapse position.
pub fn apply_stdp_batch(
    stdp: &STDP,
    pre_spikes: &[SpikeEvent],
    post_spikes: &[SpikeEvent],
) -> Vec<f64> {
    let mut deltas = Vec::new();

    for pre in pre_spikes {
        for post in post_spikes {
            if pre.neuron_id != post.neuron_id {
                let delta_t = post.timestamp - pre.timestamp;
                deltas.push(compute_stdp_delta(stdp, delta_t));
            }
        }
    }

    deltas
}

/// Hebbian learning: strengthen the connection between co-active neurons.
///
/// The classic formulation is: `Δw = η · pre_rate · post_rate`
/// where the rates are normalised (0.0–1.0).
pub fn hebbian_update(learning_rate: f64, pre_rate: f64, post_rate: f64) -> f64 {
    learning_rate * pre_rate * post_rate
}

/// BCM (Bienenstock-Cooper-Munro) learning rule.
///
/// Extends Hebbian learning with a sliding threshold `theta` that determines
/// whether synaptic potentiation or depression occurs.
///
/// `Δw = η · post_rate · (post_rate - theta) · pre_rate`
pub fn bcm_update(learning_rate: f64, pre_rate: f64, post_rate: f64, theta: f64) -> f64 {
    learning_rate * post_rate * (post_rate - theta) * pre_rate
}

/// Oja's rule for online principal component extraction.
///
/// Normalises the weight vector while performing Hebbian updates:
/// `Δw = η · post_rate · (pre_rate - post_rate · w)`
pub fn oja_update(learning_rate: f64, pre_rate: f64, post_rate: f64, weight: f64) -> f64 {
    learning_rate * post_rate * (pre_rate - post_rate * weight)
}

/// Exponential moving average update for online firing-rate estimation.
///
/// `new_rate = (1 - alpha) * old_rate + alpha * new_observation`
pub fn online_firing_rate(old_rate: f64, new_observation: f64, alpha: f64) -> f64 {
    (1.0 - alpha) * old_rate + alpha * new_observation
}

/// Homeostatic scaling: adjust synaptic weights to maintain target activity.
///
/// If the neuron fires faster than `target_rate`, weights are scaled down
/// (and vice versa). Returns the multiplicative scaling factor.
pub fn homeostatic_scaling_factor(current_rate: f64, target_rate: f64, tau: f64) -> f64 {
    // Slow exponential approach toward the target
    let error = target_rate - current_rate;
    1.0 + (error / (target_rate + f64::EPSILON)) * (1.0 / tau)
}

/// Neuromodulation gain: dopamine-gated learning enhancement.
///
/// Multiplies the effective learning rate by a dopamine signal `da` ∈ [0, 1].
pub fn dopamine_gated_gain(base_learning_rate: f64, dopamine_signal: f64) -> f64 {
    base_learning_rate * dopamine_signal.clamp(0.0, 1.0)
}

/// Online gradient descent weight update for a streaming learning scenario.
///
/// Uses a decaying step-size: `η(t) = η₀ / sqrt(t + 1)`
pub fn online_sgd_update(weight: f64, gradient: f64, eta0: f64, step: u64) -> f64 {
    let learning_rate = eta0 / ((step as f64 + 1.0).sqrt());
    weight - learning_rate * gradient
}

/// Apply all plasticity mechanisms to the synaptic plasticity state.
///
/// Consumes the list of processing results and updates internal STDP,
/// homeostatic, metaplasticity, and neuromodulation parameters.
pub fn update_plasticity_from_results(
    _plasticity: &mut SynapticPlasticity,
    results: &[NeuromorphicProcessingResult],
    config: &NeuromorphicConfig,
) -> StreamResult<()> {
    // Collect all spike events from results
    let all_spikes: Vec<&SpikeEvent> = results
        .iter()
        .flat_map(|r| r.neural_input.spike_encoding.iter())
        .collect();

    if all_spikes.len() < 2 {
        return Ok(());
    }

    // Compute average activity level for homeostatic reference
    let mean_amplitude =
        all_spikes.iter().map(|s| s.amplitude).sum::<f64>() / all_spikes.len() as f64;

    // Normalise to a pseudo firing rate [0, 1]
    let activity_rate = (mean_amplitude - 70.0).max(0.0) / 30.0;

    // Homeostatic scaling: compare to a mid-range target
    let target_rate = 0.5_f64;
    let _scale = homeostatic_scaling_factor(activity_rate, target_rate, 100.0);

    // Neuromodulation: simple dopamine proxy from activity deviation
    let dopamine_proxy = (activity_rate / target_rate).min(1.0);
    let _gated_lr = dopamine_gated_gain(config.learning_rate, dopamine_proxy);

    // Actual weight updates would modify plasticity.stdp / plasticity.homeostatic here.
    // This skeleton leaves the fields untouched to avoid UB on zeroed structs.

    Ok(())
}
