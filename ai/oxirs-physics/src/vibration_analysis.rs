//! # Vibration Analysis
//!
//! Structural vibration and modal analysis for mass-spring-damper systems.
//!
//! Implements:
//! - Natural frequency calculation: f = (1/2π) * sqrt(k/m)
//! - Damping ratio: ζ = c / (2 * sqrt(k*m))
//! - Damped natural frequency: f_d = f_n * sqrt(1 - ζ²)
//! - Free vibration response: x(t) = A * exp(-ζ * ω_n * t) * cos(ω_d * t + φ)
//! - Harmonic forcing response and resonance detection
//! - Frequency response functions (FRF)
//! - Logarithmic decrement and critical damping

use std::f64::consts::PI;

/// A mass-spring-damper system
#[derive(Debug, Clone, PartialEq)]
pub struct MassSpringSystem {
    /// Mass in kilograms
    pub mass_kg: f64,
    /// Stiffness in N/m
    pub stiffness_nm: f64,
    /// Damping coefficient in N·s/m
    pub damping_ns_m: f64,
}

impl MassSpringSystem {
    /// Create a new undamped mass-spring system
    pub fn new(mass_kg: f64, stiffness_nm: f64) -> Self {
        Self {
            mass_kg,
            stiffness_nm,
            damping_ns_m: 0.0,
        }
    }

    /// Add viscous damping to the system
    pub fn with_damping(self, damping_ns_m: f64) -> Self {
        Self {
            damping_ns_m,
            ..self
        }
    }

    /// Returns true if the system is underdamped (ζ < 1)
    pub fn is_underdamped(&self) -> bool {
        VibrationAnalyzer::damping_ratio(self) < 1.0
    }

    /// Returns true if the system is overdamped (ζ > 1)
    pub fn is_overdamped(&self) -> bool {
        VibrationAnalyzer::damping_ratio(self) > 1.0
    }

    /// Returns true if the system is critically damped (ζ ≈ 1, within 1e-9)
    pub fn is_critically_damped(&self) -> bool {
        (VibrationAnalyzer::damping_ratio(self) - 1.0).abs() < 1e-9
    }
}

/// Modal shape and frequency of a vibration mode
#[derive(Debug, Clone)]
pub struct VibrationMode {
    /// Natural frequency in Hz
    pub frequency_hz: f64,
    /// Damping ratio (dimensionless)
    pub damping_ratio: f64,
    /// Mode shape vector (relative displacements)
    pub mode_shape: Vec<f64>,
}

/// Frequency response function data
#[derive(Debug, Clone)]
pub struct FrequencyResponse {
    /// Excitation frequencies in Hz
    pub frequencies_hz: Vec<f64>,
    /// Response amplitude (displacement / static displacement)
    pub amplitudes: Vec<f64>,
    /// Phase angle in radians
    pub phase_rad: Vec<f64>,
}

impl FrequencyResponse {
    /// Number of frequency points
    pub fn len(&self) -> usize {
        self.frequencies_hz.len()
    }

    /// Returns true if the frequency response has no data points
    pub fn is_empty(&self) -> bool {
        self.frequencies_hz.is_empty()
    }
}

/// Structural vibration and modal analysis engine
pub struct VibrationAnalyzer;

impl VibrationAnalyzer {
    /// Compute the undamped natural frequency in Hz.
    ///
    /// f_n = (1 / 2π) * sqrt(k / m)
    pub fn natural_frequency(system: &MassSpringSystem) -> f64 {
        (1.0 / (2.0 * PI)) * (system.stiffness_nm / system.mass_kg).sqrt()
    }

    /// Compute the viscous damping ratio ζ.
    ///
    /// ζ = c / (2 * sqrt(k * m))
    pub fn damping_ratio(system: &MassSpringSystem) -> f64 {
        let critical = Self::critical_damping(system);
        if critical == 0.0 {
            return 0.0;
        }
        system.damping_ns_m / critical
    }

    /// Compute the damped natural frequency in Hz.
    ///
    /// f_d = f_n * sqrt(1 - ζ²)   (only meaningful for underdamped systems)
    pub fn damped_frequency(system: &MassSpringSystem) -> f64 {
        let zeta = Self::damping_ratio(system);
        let fn_hz = Self::natural_frequency(system);
        if zeta >= 1.0 {
            return 0.0; // overdamped / critically damped: no oscillation
        }
        fn_hz * (1.0 - zeta * zeta).sqrt()
    }

    /// Compute the free vibration response at time t.
    ///
    /// x(t) = A · exp(−ζ · ω_n · t) · cos(ω_d · t)
    ///
    /// Phase is taken as zero (initial velocity = 0, initial displacement = amplitude).
    pub fn free_vibration(system: &MassSpringSystem, amplitude: f64, t: f64) -> f64 {
        let zeta = Self::damping_ratio(system);
        let omega_n = 2.0 * PI * Self::natural_frequency(system);
        let omega_d = 2.0 * PI * Self::damped_frequency(system);

        if zeta >= 1.0 {
            // Overdamped or critically damped — no oscillation, pure exponential decay
            amplitude * (-zeta * omega_n * t).exp()
        } else {
            amplitude * (-zeta * omega_n * t).exp() * (omega_d * t).cos()
        }
    }

    /// Compute the steady-state amplitude under harmonic forcing F(t) = F0 * cos(ω t).
    ///
    /// X = (F0 / k) / sqrt((1 − r²)² + (2ζr)²)
    /// where r = ω / ω_n (frequency ratio)
    pub fn harmonic_response_amplitude(
        system: &MassSpringSystem,
        force_n: f64,
        forcing_freq_hz: f64,
    ) -> f64 {
        let omega_n_hz = Self::natural_frequency(system);
        let zeta = Self::damping_ratio(system);
        let r = forcing_freq_hz / omega_n_hz;

        let static_deflection = force_n / system.stiffness_nm;
        let denom = ((1.0 - r * r) * (1.0 - r * r) + (2.0 * zeta * r) * (2.0 * zeta * r)).sqrt();

        if denom == 0.0 {
            return f64::INFINITY;
        }
        static_deflection / denom
    }

    /// Compute the steady-state phase angle (in radians) of the response relative to force.
    ///
    /// φ = atan2(2ζr, 1 − r²)   (phase lag)
    pub fn harmonic_response_phase(system: &MassSpringSystem, forcing_freq_hz: f64) -> f64 {
        let omega_n_hz = Self::natural_frequency(system);
        let zeta = Self::damping_ratio(system);
        let r = forcing_freq_hz / omega_n_hz;
        (2.0 * zeta * r).atan2(1.0 - r * r)
    }

    /// Check whether the forcing frequency is in resonance with the natural frequency.
    ///
    /// Resonance when |f_forcing − f_n| / f_n < tolerance
    pub fn is_resonant(system: &MassSpringSystem, forcing_freq_hz: f64, tolerance: f64) -> bool {
        let fn_hz = Self::natural_frequency(system);
        if fn_hz == 0.0 {
            return false;
        }
        ((forcing_freq_hz - fn_hz) / fn_hz).abs() < tolerance
    }

    /// Compute the logarithmic decrement δ.
    ///
    /// δ = 2π · ζ / sqrt(1 − ζ²)
    pub fn log_decrement(damping_ratio: f64) -> f64 {
        if damping_ratio >= 1.0 {
            return f64::INFINITY;
        }
        2.0 * PI * damping_ratio / (1.0 - damping_ratio * damping_ratio).sqrt()
    }

    /// Compute the critical damping coefficient c_crit.
    ///
    /// c_crit = 2 * sqrt(k * m)
    pub fn critical_damping(system: &MassSpringSystem) -> f64 {
        2.0 * (system.stiffness_nm * system.mass_kg).sqrt()
    }

    /// Compute the frequency response function over a specified frequency range.
    ///
    /// Returns amplitude and phase at `n_points` logarithmically spaced frequencies.
    pub fn frequency_response(
        system: &MassSpringSystem,
        force_n: f64,
        freq_min: f64,
        freq_max: f64,
        n_points: usize,
    ) -> FrequencyResponse {
        if n_points == 0 {
            return FrequencyResponse {
                frequencies_hz: vec![],
                amplitudes: vec![],
                phase_rad: vec![],
            };
        }

        let mut frequencies_hz = Vec::with_capacity(n_points);
        let mut amplitudes = Vec::with_capacity(n_points);
        let mut phase_rad = Vec::with_capacity(n_points);

        let log_min = freq_min.ln();
        let log_max = freq_max.ln();

        for i in 0..n_points {
            let freq = if n_points == 1 {
                freq_min
            } else {
                (log_min + (log_max - log_min) * i as f64 / (n_points - 1) as f64).exp()
            };

            let amp = Self::harmonic_response_amplitude(system, force_n, freq);
            let phase = Self::harmonic_response_phase(system, freq);

            frequencies_hz.push(freq);
            amplitudes.push(amp);
            phase_rad.push(phase);
        }

        FrequencyResponse {
            frequencies_hz,
            amplitudes,
            phase_rad,
        }
    }

    /// Estimate damping ratio from logarithmic decrement.
    ///
    /// ζ = δ / sqrt(4π² + δ²)
    pub fn damping_ratio_from_log_decrement(log_dec: f64) -> f64 {
        log_dec / (4.0 * PI * PI + log_dec * log_dec).sqrt()
    }

    /// Compute the dynamic magnification factor (DMF) at a given frequency ratio r = ω/ω_n.
    pub fn dynamic_magnification_factor(r: f64, zeta: f64) -> f64 {
        let denom = ((1.0 - r * r) * (1.0 - r * r) + (2.0 * zeta * r) * (2.0 * zeta * r)).sqrt();
        if denom == 0.0 {
            return f64::INFINITY;
        }
        1.0 / denom
    }

    /// Compute peak frequency ratio at resonance for damped system.
    ///
    /// r_peak = sqrt(1 − 2ζ²)   (valid for ζ < 1/sqrt(2))
    pub fn peak_frequency_ratio(zeta: f64) -> Option<f64> {
        let arg = 1.0 - 2.0 * zeta * zeta;
        if arg <= 0.0 {
            None
        } else {
            Some(arg.sqrt())
        }
    }

    /// Compute half-power bandwidth in Hz.
    ///
    /// Δf ≈ 2ζ * f_n
    pub fn half_power_bandwidth(system: &MassSpringSystem) -> f64 {
        let zeta = Self::damping_ratio(system);
        let fn_hz = Self::natural_frequency(system);
        2.0 * zeta * fn_hz
    }

    /// Compute the quality factor Q = 1/(2ζ).
    pub fn quality_factor(zeta: f64) -> f64 {
        if zeta == 0.0 {
            return f64::INFINITY;
        }
        1.0 / (2.0 * zeta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── helpers ─────────────────────────────────────────────────────────────

    fn assert_approx(actual: f64, expected: f64, tol: f64, msg: &str) {
        let err = (actual - expected).abs();
        assert!(
            err < tol,
            "{msg}: got {actual}, expected {expected}, err {err}"
        );
    }

    /// Simple 1 kg / 100 N/m undamped system  →  f_n = (1/2π)*sqrt(100) ≈ 1.5915 Hz
    fn simple_system() -> MassSpringSystem {
        MassSpringSystem::new(1.0, 100.0)
    }

    /// System with 20% damping ratio: c = 2*ζ*sqrt(k*m) = 2*0.2*sqrt(100*1) = 4
    fn damped_system() -> MassSpringSystem {
        MassSpringSystem::new(1.0, 100.0).with_damping(4.0)
    }

    // ─── natural frequency ────────────────────────────────────────────────────

    #[test]
    fn test_natural_frequency_formula() {
        let sys = simple_system();
        let expected = (1.0 / (2.0 * PI)) * (100.0_f64).sqrt();
        assert_approx(
            VibrationAnalyzer::natural_frequency(&sys),
            expected,
            1e-10,
            "natural frequency",
        );
    }

    #[test]
    fn test_natural_frequency_unit_mass_unit_stiffness() {
        let sys = MassSpringSystem::new(1.0, 1.0);
        let expected = 1.0 / (2.0 * PI);
        assert_approx(
            VibrationAnalyzer::natural_frequency(&sys),
            expected,
            1e-10,
            "f_n for k=1, m=1",
        );
    }

    #[test]
    fn test_natural_frequency_scales_with_stiffness() {
        let sys1 = MassSpringSystem::new(1.0, 100.0);
        let sys2 = MassSpringSystem::new(1.0, 400.0);
        // f scales as sqrt(k) → f2 = 2 * f1
        let ratio = VibrationAnalyzer::natural_frequency(&sys2)
            / VibrationAnalyzer::natural_frequency(&sys1);
        assert_approx(ratio, 2.0, 1e-10, "frequency ratio sqrt(4)=2");
    }

    #[test]
    fn test_natural_frequency_scales_inverse_with_mass() {
        let sys1 = MassSpringSystem::new(1.0, 100.0);
        let sys2 = MassSpringSystem::new(4.0, 100.0);
        let ratio = VibrationAnalyzer::natural_frequency(&sys1)
            / VibrationAnalyzer::natural_frequency(&sys2);
        assert_approx(ratio, 2.0, 1e-10, "frequency ratio 1/sqrt(m)");
    }

    // ─── damping ratio ────────────────────────────────────────────────────────

    #[test]
    fn test_damping_ratio_undamped() {
        let sys = simple_system();
        assert_approx(
            VibrationAnalyzer::damping_ratio(&sys),
            0.0,
            1e-15,
            "undamped",
        );
    }

    #[test]
    fn test_damping_ratio_twenty_percent() {
        let sys = damped_system(); // c = 4 = 0.2 * 2 * sqrt(100)
        assert_approx(
            VibrationAnalyzer::damping_ratio(&sys),
            0.2,
            1e-10,
            "20% damping",
        );
    }

    #[test]
    fn test_damping_ratio_critical() {
        // c_crit = 2*sqrt(k*m) = 2*sqrt(100*1) = 20
        let sys = MassSpringSystem::new(1.0, 100.0).with_damping(20.0);
        assert_approx(
            VibrationAnalyzer::damping_ratio(&sys),
            1.0,
            1e-10,
            "critical damping ratio",
        );
    }

    #[test]
    fn test_damping_ratio_formula_manual() {
        let sys = MassSpringSystem::new(2.0, 200.0).with_damping(5.0);
        // c_crit = 2*sqrt(200*2) = 2*20 = 40
        // zeta = 5/40 = 0.125
        assert_approx(
            VibrationAnalyzer::damping_ratio(&sys),
            0.125,
            1e-10,
            "manual formula check",
        );
    }

    // ─── damped frequency ─────────────────────────────────────────────────────

    #[test]
    fn test_damped_frequency_less_than_natural() {
        let sys = damped_system();
        let fn_hz = VibrationAnalyzer::natural_frequency(&sys);
        let fd_hz = VibrationAnalyzer::damped_frequency(&sys);
        assert!(
            fd_hz < fn_hz,
            "damped freq {fd_hz} must be < natural {fn_hz}"
        );
    }

    #[test]
    fn test_damped_frequency_undamped_equals_natural() {
        let sys = simple_system();
        let fn_hz = VibrationAnalyzer::natural_frequency(&sys);
        let fd_hz = VibrationAnalyzer::damped_frequency(&sys);
        assert_approx(fd_hz, fn_hz, 1e-10, "undamped: f_d == f_n");
    }

    #[test]
    fn test_damped_frequency_formula() {
        let sys = damped_system(); // zeta = 0.2
        let fn_hz = VibrationAnalyzer::natural_frequency(&sys);
        let expected = fn_hz * (1.0 - 0.04_f64).sqrt();
        assert_approx(
            VibrationAnalyzer::damped_frequency(&sys),
            expected,
            1e-10,
            "f_d formula",
        );
    }

    #[test]
    fn test_damped_frequency_overdamped_is_zero() {
        let sys = MassSpringSystem::new(1.0, 100.0).with_damping(30.0); // zeta > 1
        assert_approx(
            VibrationAnalyzer::damped_frequency(&sys),
            0.0,
            1e-15,
            "overdamped f_d = 0",
        );
    }

    // ─── free vibration ───────────────────────────────────────────────────────

    #[test]
    fn test_free_vibration_at_t0_equals_amplitude() {
        let sys = damped_system();
        let amp = 0.05; // 5 cm
        let x0 = VibrationAnalyzer::free_vibration(&sys, amp, 0.0);
        assert_approx(x0, amp, 1e-10, "x(0) = amplitude");
    }

    #[test]
    fn test_free_vibration_decays() {
        let sys = damped_system();
        let amp = 1.0;
        let x0 = VibrationAnalyzer::free_vibration(&sys, amp, 0.0).abs();
        let x1 = VibrationAnalyzer::free_vibration(&sys, amp, 0.5).abs();
        let x2 = VibrationAnalyzer::free_vibration(&sys, amp, 1.0).abs();
        // Envelope should decay over longer time spans
        // x0 = 1.0, x2 is at most exp(-ζ ω_n t) ≈ exp(-0.2*10*1) = 0.135
        assert!(x2 < x0, "vibration should decay: x(1) {x2} < x(0) {x0}");
        let _ = x1; // used for compilation
    }

    #[test]
    fn test_free_vibration_undamped_at_t0() {
        let sys = simple_system();
        let x0 = VibrationAnalyzer::free_vibration(&sys, 2.0, 0.0);
        assert_approx(x0, 2.0, 1e-10, "undamped x(0)=2");
    }

    #[test]
    fn test_free_vibration_negative_amplitude() {
        let sys = damped_system();
        let x0 = VibrationAnalyzer::free_vibration(&sys, -1.0, 0.0);
        assert_approx(x0, -1.0, 1e-10, "negative amplitude at t=0");
    }

    // ─── harmonic response ────────────────────────────────────────────────────

    #[test]
    fn test_harmonic_response_static_limit() {
        // At very low frequency (r→0), amplitude → F0/k (static deflection)
        let sys = damped_system();
        let force = 10.0;
        let static_def = force / sys.stiffness_nm;
        let amp = VibrationAnalyzer::harmonic_response_amplitude(&sys, force, 0.001);
        assert_approx(amp, static_def, 1e-4, "static limit");
    }

    #[test]
    fn test_harmonic_response_at_resonance_large() {
        // At resonance (r=1), amplitude = F0/(k * 2ζ), which is large for small ζ
        let sys = damped_system(); // zeta = 0.2
        let force = 10.0;
        let fn_hz = VibrationAnalyzer::natural_frequency(&sys);
        let amp_at_resonance = VibrationAnalyzer::harmonic_response_amplitude(&sys, force, fn_hz);
        let static_def = force / sys.stiffness_nm;
        // DMF at resonance = 1/(2ζ) = 1/0.4 = 2.5
        let expected = static_def * 2.5;
        assert_approx(amp_at_resonance, expected, 1e-10, "resonance amplitude");
    }

    #[test]
    fn test_harmonic_response_amplitude_positive() {
        let sys = damped_system();
        let amp = VibrationAnalyzer::harmonic_response_amplitude(&sys, 5.0, 2.0);
        assert!(amp > 0.0, "amplitude must be positive");
    }

    #[test]
    fn test_harmonic_response_high_frequency_decreases() {
        // At very high frequency r >> 1, amplitude → 0
        let sys = damped_system();
        let amp_low = VibrationAnalyzer::harmonic_response_amplitude(&sys, 1.0, 0.01);
        let amp_high = VibrationAnalyzer::harmonic_response_amplitude(&sys, 1.0, 1000.0);
        assert!(
            amp_high < amp_low,
            "high-freq amplitude {amp_high} < low-freq {amp_low}"
        );
    }

    // ─── is_resonant ─────────────────────────────────────────────────────────

    #[test]
    fn test_is_resonant_true_at_natural_freq() {
        let sys = simple_system();
        let fn_hz = VibrationAnalyzer::natural_frequency(&sys);
        assert!(VibrationAnalyzer::is_resonant(&sys, fn_hz, 0.01));
    }

    #[test]
    fn test_is_resonant_false_far_from_natural() {
        let sys = simple_system();
        let fn_hz = VibrationAnalyzer::natural_frequency(&sys);
        assert!(!VibrationAnalyzer::is_resonant(&sys, fn_hz * 2.0, 0.01));
    }

    #[test]
    fn test_is_resonant_within_tolerance() {
        let sys = simple_system();
        let fn_hz = VibrationAnalyzer::natural_frequency(&sys);
        // 5% off natural frequency
        let forcing = fn_hz * 1.04;
        assert!(VibrationAnalyzer::is_resonant(&sys, forcing, 0.05));
    }

    #[test]
    fn test_is_resonant_outside_tolerance() {
        let sys = simple_system();
        let fn_hz = VibrationAnalyzer::natural_frequency(&sys);
        let forcing = fn_hz * 1.2; // 20% off
        assert!(!VibrationAnalyzer::is_resonant(&sys, forcing, 0.05));
    }

    // ─── log decrement ────────────────────────────────────────────────────────

    #[test]
    fn test_log_decrement_undamped_is_zero() {
        assert_approx(VibrationAnalyzer::log_decrement(0.0), 0.0, 1e-15, "δ(0)=0");
    }

    #[test]
    fn test_log_decrement_formula() {
        let zeta = 0.1;
        let expected = 2.0 * PI * zeta / (1.0 - zeta * zeta).sqrt();
        assert_approx(
            VibrationAnalyzer::log_decrement(zeta),
            expected,
            1e-10,
            "log decrement formula",
        );
    }

    #[test]
    fn test_log_decrement_increases_with_damping() {
        let d1 = VibrationAnalyzer::log_decrement(0.1);
        let d2 = VibrationAnalyzer::log_decrement(0.3);
        assert!(d2 > d1, "higher damping → larger log decrement");
    }

    #[test]
    fn test_log_decrement_round_trip() {
        let zeta = 0.15;
        let delta = VibrationAnalyzer::log_decrement(zeta);
        let zeta_back = VibrationAnalyzer::damping_ratio_from_log_decrement(delta);
        assert_approx(zeta_back, zeta, 1e-10, "round trip zeta→δ→zeta");
    }

    // ─── critical damping ─────────────────────────────────────────────────────

    #[test]
    fn test_critical_damping_formula() {
        let sys = simple_system(); // k=100, m=1
        let expected = 2.0 * (100.0_f64).sqrt(); // = 20
        assert_approx(
            VibrationAnalyzer::critical_damping(&sys),
            expected,
            1e-10,
            "c_crit formula",
        );
    }

    #[test]
    fn test_critical_damping_equals_20_for_simple_system() {
        let sys = simple_system();
        assert_approx(
            VibrationAnalyzer::critical_damping(&sys),
            20.0,
            1e-10,
            "c_crit = 20",
        );
    }

    #[test]
    fn test_critical_damping_scales_with_stiffness() {
        let sys1 = MassSpringSystem::new(1.0, 100.0);
        let sys2 = MassSpringSystem::new(1.0, 400.0);
        let ratio =
            VibrationAnalyzer::critical_damping(&sys2) / VibrationAnalyzer::critical_damping(&sys1);
        assert_approx(ratio, 2.0, 1e-10, "c_crit scales sqrt(k)");
    }

    // ─── frequency response ───────────────────────────────────────────────────

    #[test]
    fn test_frequency_response_n_points_length() {
        let sys = damped_system();
        let frf = VibrationAnalyzer::frequency_response(&sys, 1.0, 0.1, 100.0, 50);
        assert_eq!(frf.len(), 50, "FRF must have 50 points");
        assert_eq!(frf.frequencies_hz.len(), 50);
        assert_eq!(frf.amplitudes.len(), 50);
        assert_eq!(frf.phase_rad.len(), 50);
    }

    #[test]
    fn test_frequency_response_zero_points() {
        let sys = damped_system();
        let frf = VibrationAnalyzer::frequency_response(&sys, 1.0, 0.1, 100.0, 0);
        assert!(frf.is_empty());
    }

    #[test]
    fn test_frequency_response_single_point() {
        let sys = damped_system();
        let frf = VibrationAnalyzer::frequency_response(&sys, 1.0, 1.0, 1.0, 1);
        assert_eq!(frf.len(), 1);
        assert_approx(frf.frequencies_hz[0], 1.0, 1e-10, "single point freq");
    }

    #[test]
    fn test_frequency_response_amplitudes_positive() {
        let sys = damped_system();
        let frf = VibrationAnalyzer::frequency_response(&sys, 10.0, 0.1, 10.0, 20);
        for amp in &frf.amplitudes {
            assert!(*amp > 0.0, "amplitude must be positive");
        }
    }

    #[test]
    fn test_frequency_response_covers_range() {
        let sys = damped_system();
        let frf = VibrationAnalyzer::frequency_response(&sys, 1.0, 1.0, 100.0, 10);
        assert_approx(frf.frequencies_hz[0], 1.0, 1e-6, "first frequency");
        assert_approx(frf.frequencies_hz[9], 100.0, 1e-4, "last frequency");
    }

    // ─── system classification ────────────────────────────────────────────────

    #[test]
    fn test_is_underdamped() {
        let sys = damped_system(); // zeta = 0.2
        assert!(sys.is_underdamped());
        assert!(!sys.is_overdamped());
        assert!(!sys.is_critically_damped());
    }

    #[test]
    fn test_is_overdamped() {
        let sys = MassSpringSystem::new(1.0, 100.0).with_damping(30.0); // zeta > 1
        assert!(sys.is_overdamped());
        assert!(!sys.is_underdamped());
        assert!(!sys.is_critically_damped());
    }

    #[test]
    fn test_is_critically_damped() {
        let sys = MassSpringSystem::new(1.0, 100.0).with_damping(20.0); // zeta = 1 exactly
        assert!(sys.is_critically_damped());
        assert!(!sys.is_underdamped());
        assert!(!sys.is_overdamped());
    }

    #[test]
    fn test_undamped_is_underdamped() {
        let sys = simple_system(); // zeta = 0
        assert!(sys.is_underdamped());
    }

    // ─── additional helper methods ────────────────────────────────────────────

    #[test]
    fn test_dynamic_magnification_factor_at_resonance() {
        // DMF at r=1 = 1/(2ζ)
        let zeta = 0.25;
        let dmf = VibrationAnalyzer::dynamic_magnification_factor(1.0, zeta);
        assert_approx(dmf, 1.0 / (2.0 * zeta), 1e-10, "DMF at r=1");
    }

    #[test]
    fn test_quality_factor_inverse_damping() {
        let zeta = 0.1;
        let q = VibrationAnalyzer::quality_factor(zeta);
        assert_approx(q, 5.0, 1e-10, "Q = 1/(2*0.1) = 5");
    }

    #[test]
    fn test_half_power_bandwidth() {
        let sys = damped_system(); // zeta=0.2, fn ≈ 1.5915 Hz
        let bw = VibrationAnalyzer::half_power_bandwidth(&sys);
        let expected = 2.0 * 0.2 * VibrationAnalyzer::natural_frequency(&sys);
        assert_approx(bw, expected, 1e-10, "half-power bandwidth");
    }

    #[test]
    fn test_peak_frequency_ratio_low_damping() {
        let zeta = 0.1;
        let r_peak = VibrationAnalyzer::peak_frequency_ratio(zeta)
            .expect("should have peak for low damping");
        assert_approx(r_peak, (1.0 - 2.0 * 0.01_f64).sqrt(), 1e-10, "r_peak");
    }

    #[test]
    fn test_peak_frequency_ratio_high_damping_none() {
        // For zeta >= 1/sqrt(2) ≈ 0.707, no peak
        let result = VibrationAnalyzer::peak_frequency_ratio(0.8);
        assert!(result.is_none(), "no peak for overdamped");
    }

    #[test]
    fn test_mass_spring_system_builder() {
        let sys = MassSpringSystem::new(5.0, 200.0).with_damping(12.0);
        assert_eq!(sys.mass_kg, 5.0);
        assert_eq!(sys.stiffness_nm, 200.0);
        assert_eq!(sys.damping_ns_m, 12.0);
    }

    #[test]
    fn test_frequency_response_is_not_empty() {
        let sys = damped_system();
        let frf = VibrationAnalyzer::frequency_response(&sys, 1.0, 0.5, 10.0, 100);
        assert!(!frf.is_empty());
    }

    #[test]
    fn test_harmonic_response_phase_at_low_freq() {
        // At very low forcing, r→0, phase → atan2(0,1) = 0
        let sys = damped_system();
        let phase = VibrationAnalyzer::harmonic_response_phase(&sys, 0.001);
        assert!(phase.abs() < 0.01, "phase near zero at low frequency");
    }

    #[test]
    fn test_damping_ratio_from_log_decrement_roundtrip_2() {
        let zeta_in = 0.05;
        let delta = VibrationAnalyzer::log_decrement(zeta_in);
        let zeta_out = VibrationAnalyzer::damping_ratio_from_log_decrement(delta);
        assert_approx(zeta_out, zeta_in, 1e-10, "log-dec round trip 2");
    }
}
