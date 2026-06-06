//! Advanced validation algorithms: entropy, angular momentum, Noether symmetry.
//!
//! Contains `EntropyConservationChecker`, `AngularMomentumChecker`,
//! and `NoetherSymmetryValidator`.

use super::PhysState;
use crate::conservation::checkers_impl::{EnergyConservationChecker, MomentumConservationChecker};
use crate::conservation::checkers_types::{
    ConservationReport, ConservationViolationDetail, NoetherCheckResult, PhysicalSymmetry,
};

// ──────────────────────────────────────────────────────────────────────────────
// EntropyConservationChecker
// ──────────────────────────────────────────────────────────────────────────────

/// Validates the Second Law of Thermodynamics: entropy must not decrease.
///
/// For a closed system, `S_final >= S_initial - tolerance` (non-decrease).
/// The Clausius inequality `ΔS >= Q/T` is checked when heat flow `Q` and
/// temperature `T` are both present in the state.
pub struct EntropyConservationChecker {
    /// Absolute tolerance for entropy non-decrease (J/K).
    pub abs_tolerance: f64,
    /// If true, also validate the Clausius inequality when Q and T are available.
    pub check_clausius: bool,
}

impl EntropyConservationChecker {
    /// Create a new entropy checker.
    pub fn new(abs_tolerance: f64) -> Self {
        Self {
            abs_tolerance,
            check_clausius: true,
        }
    }

    /// Disable Clausius inequality check (entropy-only mode).
    pub fn without_clausius(mut self) -> Self {
        self.check_clausius = false;
        self
    }

    /// Create with tight default tolerance (1 μJ/K).
    pub fn default_tolerances() -> Self {
        Self::new(1e-6)
    }

    /// Check entropy monotonicity between two states.
    ///
    /// A violation occurs when `S_final - S_initial < -abs_tolerance`.
    pub fn check_pair(
        &self,
        initial: &PhysState,
        final_state: &PhysState,
        step_index: Option<usize>,
    ) -> ConservationReport {
        let mut report = ConservationReport::new("EntropyConservationChecker");
        report.states_checked = 1;

        let s_i = initial.get("entropy").unwrap_or(0.0);
        let s_f = final_state.get("entropy").unwrap_or(0.0);
        let delta = s_f - s_i;
        report.track_change(delta.abs());

        // Second Law: entropy must not decrease
        if delta < -self.abs_tolerance {
            report.add_violation(ConservationViolationDetail::new(
                "Second Law of Thermodynamics",
                "entropy",
                s_i,
                s_f,
                self.abs_tolerance,
                step_index,
            ));
        }

        // Clausius inequality: ΔS >= Q/T when heat and temperature are available
        if self.check_clausius {
            let q = final_state
                .get("heat_flow")
                .or_else(|| initial.get("heat_flow"))
                .unwrap_or(0.0);
            let t = final_state
                .get("temperature")
                .or_else(|| initial.get("temperature"))
                .unwrap_or(0.0);
            if t > 1e-10 {
                // Clausius requires ΔS >= Q/T
                let q_over_t = q / t;
                if delta < q_over_t - self.abs_tolerance {
                    report.add_violation(ConservationViolationDetail::new(
                        "Clausius Inequality (ΔS ≥ Q/T)",
                        "entropy_vs_heat",
                        q_over_t,
                        delta,
                        self.abs_tolerance,
                        step_index,
                    ));
                }
            }
        }

        report
    }

    /// Check entropy monotonicity across a full trajectory.
    pub fn check_trajectory(&self, states: &[PhysState]) -> ConservationReport {
        let mut report = ConservationReport::new("EntropyConservationChecker");
        if states.len() < 2 {
            report.states_checked = states.len();
            return report;
        }
        report.states_checked = states.len() - 1;

        for (idx, window) in states.windows(2).enumerate() {
            let pair_report = self.check_pair(&window[0], &window[1], Some(idx));
            report.track_change(pair_report.max_absolute_change);
            for v in pair_report.violations {
                report.add_violation(v);
            }
        }
        report
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// AngularMomentumChecker
// ──────────────────────────────────────────────────────────────────────────────

/// Validates conservation of angular momentum in 3D.
///
/// Angular momentum `L = (Lx, Ly, Lz)` is conserved when `τ = dL/dt ≈ 0`
/// (no external torque). Both conservation (|ΔL| ≤ tolerance) and torque
/// consistency (when torque is given) are checked.
pub struct AngularMomentumChecker {
    /// Absolute tolerance per component (kg·m²/s).
    pub abs_tolerance: f64,
    /// Relative tolerance (fraction of initial magnitude).
    pub rel_tolerance: f64,
    /// If true, check `τ = ΔL/Δt` consistency when torque and dt are present.
    pub check_torque_consistency: bool,
}

impl AngularMomentumChecker {
    /// Create a new checker with given absolute and relative tolerances.
    pub fn new(abs_tolerance: f64, rel_tolerance: f64) -> Self {
        Self {
            abs_tolerance,
            rel_tolerance,
            check_torque_consistency: true,
        }
    }

    /// Default tolerances (1e-6 absolute, 0.1% relative).
    pub fn default_tolerances() -> Self {
        Self::new(1e-6, 1e-3)
    }

    /// Disable torque consistency check.
    pub fn without_torque_check(mut self) -> Self {
        self.check_torque_consistency = false;
        self
    }

    fn component_magnitude(lx: f64, ly: f64, lz: f64) -> f64 {
        (lx * lx + ly * ly + lz * lz).sqrt()
    }

    /// Check angular momentum conservation between two states.
    pub fn check_pair(
        &self,
        initial: &PhysState,
        final_state: &PhysState,
        step_index: Option<usize>,
    ) -> ConservationReport {
        let mut report = ConservationReport::new("AngularMomentumChecker");
        report.states_checked = 1;

        let (lx_i, ly_i, lz_i) = (
            initial.get("angular_momentum_x").unwrap_or(0.0),
            initial.get("angular_momentum_y").unwrap_or(0.0),
            initial.get("angular_momentum_z").unwrap_or(0.0),
        );
        let (lx_f, ly_f, lz_f) = (
            final_state.get("angular_momentum_x").unwrap_or(0.0),
            final_state.get("angular_momentum_y").unwrap_or(0.0),
            final_state.get("angular_momentum_z").unwrap_or(0.0),
        );

        let mag_initial = Self::component_magnitude(lx_i, ly_i, lz_i);

        // Check each component
        for (name, l_i, l_f) in [("Lx", lx_i, lx_f), ("Ly", ly_i, ly_f), ("Lz", lz_i, lz_f)] {
            let delta = (l_f - l_i).abs();
            report.track_change(delta);

            let abs_violation = delta > self.abs_tolerance;
            let rel_violation = mag_initial > 1e-300 && delta / mag_initial > self.rel_tolerance;

            if abs_violation || rel_violation {
                report.add_violation(ConservationViolationDetail::new(
                    "Angular Momentum Conservation",
                    name,
                    l_i,
                    l_f,
                    self.abs_tolerance,
                    step_index,
                ));
            }
        }

        // Torque consistency: τ = ΔL/Δt
        if self.check_torque_consistency {
            let dt = final_state
                .get("dt")
                .or_else(|| initial.get("dt"))
                .unwrap_or(0.0);
            if dt > 1e-300 {
                for (name, torque_key, l_i, l_f) in [
                    ("Lx", "torque_x", lx_i, lx_f),
                    ("Ly", "torque_y", ly_i, ly_f),
                    ("Lz", "torque_z", lz_i, lz_f),
                ] {
                    let tau = final_state
                        .get(torque_key)
                        .or_else(|| initial.get(torque_key))
                        .unwrap_or(0.0);
                    let expected_delta = tau * dt;
                    let actual_delta = l_f - l_i;
                    let discrepancy = (actual_delta - expected_delta).abs();
                    if discrepancy > self.abs_tolerance {
                        report.add_violation(ConservationViolationDetail::new(
                            format!("Torque Consistency (τ = ΔL/Δt) — {name}"),
                            torque_key,
                            expected_delta,
                            actual_delta,
                            self.abs_tolerance,
                            step_index,
                        ));
                    }
                }
            }
        }

        report
    }

    /// Check angular momentum conservation across a full trajectory.
    pub fn check_trajectory(&self, states: &[PhysState]) -> ConservationReport {
        let mut report = ConservationReport::new("AngularMomentumChecker");
        if states.len() < 2 {
            report.states_checked = states.len();
            return report;
        }
        report.states_checked = states.len() - 1;

        for (idx, window) in states.windows(2).enumerate() {
            let pair_report = self.check_pair(&window[0], &window[1], Some(idx));
            report.track_change(pair_report.max_absolute_change);
            for v in pair_report.violations {
                report.add_violation(v);
            }
        }
        report
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// NoetherSymmetryValidator
// ──────────────────────────────────────────────────────────────────────────────

/// Maps physical symmetries to their conserved quantities via Noether's theorem.
///
/// Noether's theorem states that every continuous symmetry of the action
/// corresponds to a conserved quantity:
/// - **Time translation symmetry** → Energy conservation
/// - **Spatial translation symmetry** → Linear momentum conservation
/// - **Rotational symmetry** → Angular momentum conservation
///
/// This validator checks whether a given state trajectory is consistent
/// with the conserved quantities expected for declared symmetries.
#[derive(Debug, Clone)]
pub struct NoetherSymmetryValidator {
    /// Declared symmetries present in the system.
    pub symmetries: Vec<PhysicalSymmetry>,
    /// Absolute tolerance used when checking conserved quantities.
    pub abs_tolerance: f64,
}

impl NoetherSymmetryValidator {
    /// Create a validator for the given symmetries.
    pub fn new(symmetries: Vec<PhysicalSymmetry>, abs_tolerance: f64) -> Self {
        Self {
            symmetries,
            abs_tolerance,
        }
    }

    /// Validate all declared symmetries against a state trajectory.
    ///
    /// Returns one [`NoetherCheckResult`] per declared symmetry.
    pub fn validate(&self, states: &[PhysState]) -> Vec<NoetherCheckResult> {
        self.symmetries
            .iter()
            .map(|&sym| {
                let report = match sym {
                    PhysicalSymmetry::TimeTranslation => {
                        EnergyConservationChecker::new(self.abs_tolerance, f64::MAX)
                            .check_trajectory(states)
                    }
                    PhysicalSymmetry::SpatialTranslation => {
                        MomentumConservationChecker::new(self.abs_tolerance, f64::MAX)
                            .check_trajectory(states)
                    }
                    PhysicalSymmetry::Rotation => {
                        AngularMomentumChecker::new(self.abs_tolerance, f64::MAX)
                            .check_trajectory(states)
                    }
                };
                let conserved = report.passed;
                NoetherCheckResult {
                    symmetry: sym,
                    conserved,
                    report,
                }
            })
            .collect()
    }

    /// Returns `true` if all declared symmetries have their conserved quantity
    /// verified in the trajectory.
    pub fn all_conserved(&self, states: &[PhysState]) -> bool {
        self.validate(states).iter().all(|r| r.conserved)
    }
}
