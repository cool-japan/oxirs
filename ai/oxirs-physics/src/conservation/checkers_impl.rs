//! Main conservation checker implementations.
//!
//! Contains `EnergyConservationChecker`, `MomentumConservationChecker`,
//! `MassConservationChecker`, and `ConservationSuite`.

use super::PhysState;
use crate::conservation::checkers_types::{ConservationReport, ConservationViolationDetail};

// ──────────────────────────────────────────────────────────────────────────────
// EnergyConservationChecker
// ──────────────────────────────────────────────────────────────────────────────

/// Validates energy conservation across a sequence of states.
///
/// Two tolerance modes are supported:
/// - **Absolute** (`abs_tolerance`): `|ΔE| ≤ abs_tolerance`
/// - **Relative** (`rel_tolerance`): `|ΔE / E_initial| ≤ rel_tolerance`
///
/// A violation occurs when either threshold is exceeded.
pub struct EnergyConservationChecker {
    /// Absolute energy tolerance (Joules).
    pub abs_tolerance: f64,
    /// Relative energy tolerance (dimensionless fraction).
    pub rel_tolerance: f64,
}

impl EnergyConservationChecker {
    /// Create a new checker with given absolute and relative tolerances.
    pub fn new(abs_tolerance: f64, rel_tolerance: f64) -> Self {
        Self {
            abs_tolerance,
            rel_tolerance,
        }
    }

    /// Create with typical physics defaults (1 mJ absolute, 0.1% relative).
    pub fn default_tolerances() -> Self {
        Self::new(1e-3, 1e-3)
    }

    /// Check energy conservation between two states.
    pub fn check_pair(
        &self,
        initial: &PhysState,
        final_state: &PhysState,
        step_index: Option<usize>,
    ) -> ConservationReport {
        let mut report = ConservationReport::new("EnergyConservationChecker");
        report.states_checked = 1;

        let e_i = Self::total_energy(initial);
        let e_f = Self::total_energy(final_state);
        let delta = (e_f - e_i).abs();

        report.track_change(delta);

        let rel_violation = e_i.abs() > 1e-300 && delta / e_i.abs() > self.rel_tolerance;
        let abs_violation = delta > self.abs_tolerance;

        if abs_violation || rel_violation {
            report.add_violation(ConservationViolationDetail::new(
                "Energy Conservation",
                "total_energy",
                e_i,
                e_f,
                self.abs_tolerance,
                step_index,
            ));
        }

        report
    }

    /// Check energy conservation across a full trajectory (consecutive pairs).
    pub fn check_trajectory(&self, states: &[PhysState]) -> ConservationReport {
        let mut report = ConservationReport::new("EnergyConservationChecker");
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

    fn total_energy(state: &PhysState) -> f64 {
        if let Some(e) = state.get("total_energy") {
            return e;
        }
        state.get("kinetic_energy").unwrap_or(0.0) + state.get("potential_energy").unwrap_or(0.0)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// MomentumConservationChecker
// ──────────────────────────────────────────────────────────────────────────────

/// Validates linear and angular momentum conservation with per-component
/// violation details.
pub struct MomentumConservationChecker {
    /// Absolute momentum tolerance (kg⋅m/s).
    pub abs_tolerance: f64,
    /// Relative momentum tolerance (dimensionless fraction).
    pub rel_tolerance: f64,
    /// Check angular momentum (`angular_momentum` key).
    pub check_angular: bool,
}

impl MomentumConservationChecker {
    /// Create a new checker.
    pub fn new(abs_tolerance: f64, rel_tolerance: f64) -> Self {
        Self {
            abs_tolerance,
            rel_tolerance,
            check_angular: true,
        }
    }

    /// Create with typical defaults (1e-6 kg⋅m/s absolute, 0.01% relative).
    pub fn default_tolerances() -> Self {
        Self::new(1e-6, 1e-4)
    }

    /// Disable angular momentum checking.
    pub fn without_angular(mut self) -> Self {
        self.check_angular = false;
        self
    }

    /// Check momentum conservation between two states, returning a detailed report.
    pub fn check_pair(
        &self,
        initial: &PhysState,
        final_state: &PhysState,
        step_index: Option<usize>,
    ) -> ConservationReport {
        let mut report = ConservationReport::new("MomentumConservationChecker");
        report.states_checked = 1;

        let components = &["momentum_x", "momentum_y", "momentum_z"];
        for &comp in components {
            let p_i = initial.get(comp).unwrap_or(0.0);
            let p_f = final_state.get(comp).unwrap_or(0.0);
            let delta = (p_f - p_i).abs();

            report.track_change(delta);

            let rel_violation = p_i.abs() > 1e-300 && delta / p_i.abs() > self.rel_tolerance;
            let abs_violation = delta > self.abs_tolerance;

            if abs_violation || rel_violation {
                report.add_violation(ConservationViolationDetail::new(
                    "Momentum Conservation",
                    comp,
                    p_i,
                    p_f,
                    self.abs_tolerance,
                    step_index,
                ));
            }
        }

        if self.check_angular {
            let l_i = initial.get("angular_momentum").unwrap_or(0.0);
            let l_f = final_state.get("angular_momentum").unwrap_or(0.0);
            let delta = (l_f - l_i).abs();

            report.track_change(delta);

            let rel_violation = l_i.abs() > 1e-300 && delta / l_i.abs() > self.rel_tolerance;
            let abs_violation = delta > self.abs_tolerance;

            if abs_violation || rel_violation {
                report.add_violation(ConservationViolationDetail::new(
                    "Momentum Conservation",
                    "angular_momentum",
                    l_i,
                    l_f,
                    self.abs_tolerance,
                    step_index,
                ));
            }
        }

        report
    }

    /// Check momentum conservation across a full trajectory.
    pub fn check_trajectory(&self, states: &[PhysState]) -> ConservationReport {
        let mut report = ConservationReport::new("MomentumConservationChecker");
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
// MassConservationChecker
// ──────────────────────────────────────────────────────────────────────────────

/// Validates mass conservation for fluid simulations.
///
/// Supports:
/// - **Single-species**: checks `total_mass` or `mass` key.
/// - **Multi-species**: checks individual species masses and their sum.
pub struct MassConservationChecker {
    /// Absolute mass tolerance (kg).
    pub abs_tolerance: f64,
    /// Relative mass tolerance (dimensionless fraction).
    pub rel_tolerance: f64,
    /// Species keys to check individually (for multi-species flows).
    pub species_keys: Vec<String>,
}

impl MassConservationChecker {
    /// Create a single-species checker.
    pub fn new(abs_tolerance: f64, rel_tolerance: f64) -> Self {
        Self {
            abs_tolerance,
            rel_tolerance,
            species_keys: Vec::new(),
        }
    }

    /// Create with typical defaults (1e-9 kg absolute, 1e-6 relative).
    pub fn default_tolerances() -> Self {
        Self::new(1e-9, 1e-6)
    }

    /// Add species keys for multi-species mass balance tracking.
    pub fn with_species(mut self, keys: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.species_keys = keys.into_iter().map(Into::into).collect();
        self
    }

    /// Check mass conservation between two states.
    pub fn check_pair(
        &self,
        initial: &PhysState,
        final_state: &PhysState,
        step_index: Option<usize>,
    ) -> ConservationReport {
        let mut report = ConservationReport::new("MassConservationChecker");
        report.states_checked = 1;

        // Total mass
        let m_i = initial
            .get("total_mass")
            .or_else(|| initial.get("mass"))
            .unwrap_or(0.0);
        let m_f = final_state
            .get("total_mass")
            .or_else(|| final_state.get("mass"))
            .unwrap_or(0.0);
        let delta = (m_f - m_i).abs();

        report.track_change(delta);

        let rel_violation = m_i.abs() > 1e-300 && delta / m_i.abs() > self.rel_tolerance;
        let abs_violation = delta > self.abs_tolerance;

        if abs_violation || rel_violation {
            report.add_violation(ConservationViolationDetail::new(
                "Mass Conservation",
                "total_mass",
                m_i,
                m_f,
                self.abs_tolerance,
                step_index,
            ));
        }

        // Per-species check
        for key in &self.species_keys {
            let s_i = initial.get(key).unwrap_or(0.0);
            let s_f = final_state.get(key).unwrap_or(0.0);
            let sp_delta = (s_f - s_i).abs();

            report.track_change(sp_delta);

            let sp_rel = s_i.abs() > 1e-300 && sp_delta / s_i.abs() > self.rel_tolerance;
            let sp_abs = sp_delta > self.abs_tolerance;

            if sp_abs || sp_rel {
                report.add_violation(ConservationViolationDetail::new(
                    "Mass Conservation",
                    key.as_str(),
                    s_i,
                    s_f,
                    self.abs_tolerance,
                    step_index,
                ));
            }
        }

        report
    }

    /// Check mass conservation across a full trajectory.
    pub fn check_trajectory(&self, states: &[PhysState]) -> ConservationReport {
        let mut report = ConservationReport::new("MassConservationChecker");
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
// ConservationSuite
// ──────────────────────────────────────────────────────────────────────────────

/// Runs multiple conservation checkers over the same trajectory and
/// produces a combined report.
pub struct ConservationSuite {
    energy_checker: Option<EnergyConservationChecker>,
    momentum_checker: Option<MomentumConservationChecker>,
    mass_checker: Option<MassConservationChecker>,
}

impl ConservationSuite {
    /// Create an empty suite.
    pub fn new() -> Self {
        Self {
            energy_checker: None,
            momentum_checker: None,
            mass_checker: None,
        }
    }

    /// Enable energy checking.
    pub fn with_energy(mut self, checker: EnergyConservationChecker) -> Self {
        self.energy_checker = Some(checker);
        self
    }

    /// Enable momentum checking.
    pub fn with_momentum(mut self, checker: MomentumConservationChecker) -> Self {
        self.momentum_checker = Some(checker);
        self
    }

    /// Enable mass checking.
    pub fn with_mass(mut self, checker: MassConservationChecker) -> Self {
        self.mass_checker = Some(checker);
        self
    }

    /// Run all enabled checkers over a trajectory.
    pub fn check_trajectory(&self, states: &[PhysState]) -> Vec<ConservationReport> {
        let mut reports = Vec::new();
        if let Some(ref checker) = self.energy_checker {
            reports.push(checker.check_trajectory(states));
        }
        if let Some(ref checker) = self.momentum_checker {
            reports.push(checker.check_trajectory(states));
        }
        if let Some(ref checker) = self.mass_checker {
            reports.push(checker.check_trajectory(states));
        }
        reports
    }

    /// True if all checkers pass.
    pub fn all_pass(&self, states: &[PhysState]) -> bool {
        self.check_trajectory(states).iter().all(|r| r.passed)
    }
}

impl Default for ConservationSuite {
    fn default() -> Self {
        Self::new()
    }
}
