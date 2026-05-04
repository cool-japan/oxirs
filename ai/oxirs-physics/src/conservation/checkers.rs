//! Enhanced Conservation Law Checkers
//!
//! Provides high-level, structured checker types that wrap the core
//! `ConservationLaw` implementations with detailed violation reporting,
//! multi-step trajectory analysis, and tolerance management.
//!
//! ## Checker types
//!
//! - [`EnergyConservationChecker`]: Validates energy conservation across a
//!   sequence of states with configurable relative / absolute tolerances.
//! - [`MomentumConservationChecker`]: Validates linear and angular momentum
//!   conservation per component, with per-component violation details.
//! - [`MassConservationChecker`]: Validates mass conservation for fluid
//!   simulations, supporting multi-species mass balance.
//! - [`ConservationReport`]: Aggregated report from all checkers.

use super::PhysState;
use serde::{Deserialize, Serialize};

// ──────────────────────────────────────────────────────────────────────────────
// ConservationViolationDetail
// ──────────────────────────────────────────────────────────────────────────────

/// Severity level of a conservation violation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ViolationSeverity {
    /// Within 10× the configured tolerance.
    Warning,
    /// Outside 10× the configured tolerance.
    Critical,
}

/// Detailed information about a single conservation law violation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConservationViolationDetail {
    /// Human-readable name of the conservation law violated.
    pub law_name: String,
    /// Description of the quantity that was not conserved.
    pub quantity: String,
    /// Value at the initial (reference) state.
    pub initial_value: f64,
    /// Value at the final state.
    pub final_value: f64,
    /// Absolute change: `final_value - initial_value`.
    pub absolute_change: f64,
    /// Relative change: `|absolute_change / initial_value|` (NaN if initial = 0).
    pub relative_change: f64,
    /// Configured absolute tolerance.
    pub tolerance: f64,
    /// Whether the violation is critical (>10× tolerance) or a warning.
    pub severity: ViolationSeverity,
    /// Step index (in a trajectory) where the violation occurred, if known.
    pub step_index: Option<usize>,
}

impl ConservationViolationDetail {
    fn new(
        law_name: impl Into<String>,
        quantity: impl Into<String>,
        initial_value: f64,
        final_value: f64,
        tolerance: f64,
        step_index: Option<usize>,
    ) -> Self {
        let absolute_change = final_value - initial_value;
        let relative_change = if initial_value.abs() > 1e-300 {
            (absolute_change / initial_value).abs()
        } else {
            f64::NAN
        };
        let severity = if absolute_change.abs() > 10.0 * tolerance {
            ViolationSeverity::Critical
        } else {
            ViolationSeverity::Warning
        };
        Self {
            law_name: law_name.into(),
            quantity: quantity.into(),
            initial_value,
            final_value,
            absolute_change,
            relative_change,
            tolerance,
            severity,
            step_index,
        }
    }

    /// True if the violation exceeds the critical threshold (>10× tolerance).
    pub fn is_critical(&self) -> bool {
        self.severity == ViolationSeverity::Critical
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// ConservationReport
// ──────────────────────────────────────────────────────────────────────────────

/// Aggregated conservation check report produced by a checker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConservationReport {
    /// Name of the checker that produced this report.
    pub checker_name: String,
    /// True if all conservation laws were satisfied.
    pub passed: bool,
    /// Detailed violations (empty when `passed` is true).
    pub violations: Vec<ConservationViolationDetail>,
    /// Number of state pairs checked.
    pub states_checked: usize,
    /// Maximum absolute change observed across all quantities and steps.
    pub max_absolute_change: f64,
}

impl ConservationReport {
    fn new(checker_name: impl Into<String>) -> Self {
        Self {
            checker_name: checker_name.into(),
            passed: true,
            violations: Vec::new(),
            states_checked: 0,
            max_absolute_change: 0.0,
        }
    }

    fn add_violation(&mut self, detail: ConservationViolationDetail) {
        self.max_absolute_change = self.max_absolute_change.max(detail.absolute_change.abs());
        self.violations.push(detail);
        self.passed = false;
    }

    fn track_change(&mut self, change: f64) {
        self.max_absolute_change = self.max_absolute_change.max(change.abs());
    }

    /// True if any violation is critical.
    pub fn has_critical_violations(&self) -> bool {
        self.violations.iter().any(|v| v.is_critical())
    }

    /// Return all critical violations.
    pub fn critical_violations(&self) -> impl Iterator<Item = &ConservationViolationDetail> {
        self.violations.iter().filter(|v| v.is_critical())
    }

    /// Return a one-line summary.
    pub fn summary(&self) -> String {
        if self.passed {
            format!(
                "{}: PASS ({} states checked)",
                self.checker_name, self.states_checked
            )
        } else {
            format!(
                "{}: FAIL ({} violations, max |ΔQ|={:.3e})",
                self.checker_name,
                self.violations.len(),
                self.max_absolute_change
            )
        }
    }
}

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

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::PhysState;
    use super::*;

    fn state_with(pairs: &[(&str, f64)]) -> PhysState {
        let mut s = PhysState::new();
        for &(k, v) in pairs {
            s.set(k, v);
        }
        s
    }

    // ── EnergyConservationChecker ─────────────────────────────────────────────

    #[test]
    fn test_energy_checker_pass() {
        let checker = EnergyConservationChecker::new(1.0, 0.01);
        let s0 = state_with(&[("total_energy", 1000.0)]);
        let s1 = state_with(&[("total_energy", 1000.5)]);
        let report = checker.check_pair(&s0, &s1, None);
        assert!(report.passed, "expected pass");
        assert!(report.violations.is_empty());
    }

    #[test]
    fn test_energy_checker_fail_absolute() {
        let checker = EnergyConservationChecker::new(0.1, 1.0);
        let s0 = state_with(&[("total_energy", 100.0)]);
        let s1 = state_with(&[("total_energy", 200.0)]);
        let report = checker.check_pair(&s0, &s1, None);
        assert!(!report.passed, "expected failure");
        assert_eq!(report.violations.len(), 1);
    }

    #[test]
    fn test_energy_checker_fail_relative() {
        let checker = EnergyConservationChecker::new(1000.0, 0.001); // tight relative
        let s0 = state_with(&[("total_energy", 1_000_000.0)]);
        let s1 = state_with(&[("total_energy", 1_005_000.0)]); // 0.5% change
        let report = checker.check_pair(&s0, &s1, None);
        assert!(!report.passed, "expected relative violation");
    }

    #[test]
    fn test_energy_checker_trajectory_pass() {
        let checker = EnergyConservationChecker::new(1.0, 0.01);
        let states: Vec<PhysState> = (0..5)
            .map(|i| state_with(&[("total_energy", 100.0 + i as f64 * 0.1)]))
            .collect();
        let report = checker.check_trajectory(&states);
        assert!(report.passed, "small drift should pass");
        assert_eq!(report.states_checked, 4);
    }

    #[test]
    fn test_energy_checker_trajectory_fail() {
        let checker = EnergyConservationChecker::new(0.5, 0.001);
        let states = vec![
            state_with(&[("total_energy", 100.0)]),
            state_with(&[("total_energy", 100.1)]),
            state_with(&[("total_energy", 150.0)]), // big jump
        ];
        let report = checker.check_trajectory(&states);
        assert!(!report.passed);
        assert!(!report.violations.is_empty());
    }

    #[test]
    fn test_energy_checker_violation_detail() {
        let checker = EnergyConservationChecker::new(0.1, 1.0);
        let s0 = state_with(&[("total_energy", 100.0)]);
        let s1 = state_with(&[("total_energy", 200.0)]);
        let report = checker.check_pair(&s0, &s1, Some(3));
        let v = &report.violations[0];
        assert_eq!(v.step_index, Some(3));
        assert!((v.absolute_change - 100.0).abs() < 1e-10);
        assert_eq!(v.law_name, "Energy Conservation");
    }

    #[test]
    fn test_energy_checker_critical_vs_warning() {
        let checker = EnergyConservationChecker::new(1.0, 1.0);
        // 5× tolerance → Warning
        let s0 = state_with(&[("total_energy", 100.0)]);
        let s1 = state_with(&[("total_energy", 105.0)]);
        let r1 = checker.check_pair(&s0, &s1, None);
        if !r1.violations.is_empty() {
            assert_eq!(r1.violations[0].severity, ViolationSeverity::Warning);
        }
        // 50× tolerance → Critical
        let s2 = state_with(&[("total_energy", 150.0)]);
        let r2 = checker.check_pair(&s0, &s2, None);
        assert!(!r2.violations.is_empty());
        assert_eq!(r2.violations[0].severity, ViolationSeverity::Critical);
    }

    #[test]
    fn test_energy_checker_summary_pass() {
        let checker = EnergyConservationChecker::new(1.0, 0.01);
        let s = state_with(&[("total_energy", 100.0)]);
        let report = checker.check_pair(&s, &s, None);
        assert!(report.summary().contains("PASS"));
    }

    #[test]
    fn test_energy_checker_summary_fail() {
        let checker = EnergyConservationChecker::new(0.1, 0.001);
        let s0 = state_with(&[("total_energy", 100.0)]);
        let s1 = state_with(&[("total_energy", 200.0)]);
        let report = checker.check_pair(&s0, &s1, None);
        assert!(report.summary().contains("FAIL"));
    }

    // ── MomentumConservationChecker ───────────────────────────────────────────

    #[test]
    fn test_momentum_checker_pass() {
        let checker = MomentumConservationChecker::new(1e-4, 0.01);
        let s0 = state_with(&[
            ("momentum_x", 10.0),
            ("momentum_y", 5.0),
            ("momentum_z", 0.0),
        ]);
        let s1 = s0.clone();
        let report = checker.check_pair(&s0, &s1, None);
        assert!(report.passed);
    }

    #[test]
    fn test_momentum_checker_fail_component() {
        let checker = MomentumConservationChecker::new(0.01, 1.0);
        let s0 = state_with(&[
            ("momentum_x", 10.0),
            ("momentum_y", 0.0),
            ("momentum_z", 0.0),
        ]);
        let s1 = state_with(&[
            ("momentum_x", 20.0),
            ("momentum_y", 0.0),
            ("momentum_z", 0.0),
        ]);
        let report = checker.check_pair(&s0, &s1, None);
        assert!(!report.passed);
        assert!(report.violations.iter().any(|v| v.quantity == "momentum_x"));
    }

    #[test]
    fn test_momentum_checker_angular_violation() {
        let checker = MomentumConservationChecker::new(1e-4, 0.01);
        let s0 = state_with(&[("angular_momentum", 5.0)]);
        let s1 = state_with(&[("angular_momentum", 10.0)]);
        let report = checker.check_pair(&s0, &s1, None);
        assert!(!report.passed);
        assert!(report
            .violations
            .iter()
            .any(|v| v.quantity == "angular_momentum"));
    }

    #[test]
    fn test_momentum_checker_without_angular() {
        let checker = MomentumConservationChecker::new(1e-4, 0.01).without_angular();
        let s0 = state_with(&[("angular_momentum", 5.0)]);
        let s1 = state_with(&[("angular_momentum", 100.0)]); // large angular change, but disabled
        let report = checker.check_pair(&s0, &s1, None);
        // Only linear checked (all zero → pass)
        assert!(report.passed);
    }

    #[test]
    fn test_momentum_checker_trajectory() {
        let checker = MomentumConservationChecker::new(1e-6, 1e-6);
        let states: Vec<PhysState> = vec![
            state_with(&[("momentum_x", 10.0)]),
            state_with(&[("momentum_x", 10.0 + 1e-8)]),
            state_with(&[("momentum_x", 10.0 + 2e-8)]),
        ];
        let report = checker.check_trajectory(&states);
        assert!(
            report.passed,
            "tiny drift should pass: {:?}",
            report.violations
        );
    }

    // ── MassConservationChecker ───────────────────────────────────────────────

    #[test]
    fn test_mass_checker_pass() {
        let checker = MassConservationChecker::new(1e-9, 1e-6);
        let s0 = state_with(&[("total_mass", 1.0)]);
        let s1 = state_with(&[("total_mass", 1.0 + 1e-12)]);
        let report = checker.check_pair(&s0, &s1, None);
        assert!(report.passed);
    }

    #[test]
    fn test_mass_checker_fail() {
        let checker = MassConservationChecker::new(1e-6, 1e-4);
        let s0 = state_with(&[("total_mass", 1.0)]);
        let s1 = state_with(&[("total_mass", 2.0)]);
        let report = checker.check_pair(&s0, &s1, None);
        assert!(!report.passed);
        assert!(report.violations.iter().any(|v| v.quantity == "total_mass"));
    }

    #[test]
    fn test_mass_checker_multi_species() {
        let checker =
            MassConservationChecker::new(1e-9, 1e-6).with_species(["mass_h2o", "mass_n2"]);
        let s0 = state_with(&[("mass_h2o", 0.5), ("mass_n2", 0.5)]);
        let s1 = state_with(&[("mass_h2o", 1.0), ("mass_n2", 0.5)]); // h2o doubles!
        let report = checker.check_pair(&s0, &s1, None);
        assert!(!report.passed, "species mass change should be detected");
        assert!(report.violations.iter().any(|v| v.quantity == "mass_h2o"));
    }

    #[test]
    fn test_mass_checker_trajectory() {
        let checker = MassConservationChecker::new(1e-3, 0.01);
        let states: Vec<PhysState> = (0..4)
            .map(|i| state_with(&[("mass", 10.0 + i as f64 * 1e-5)]))
            .collect();
        let report = checker.check_trajectory(&states);
        assert!(report.passed, "tiny drift should pass");
    }

    // ── ConservationSuite ─────────────────────────────────────────────────────

    #[test]
    fn test_conservation_suite_all_pass() {
        let suite = ConservationSuite::new()
            .with_energy(EnergyConservationChecker::new(1.0, 0.01))
            .with_momentum(MomentumConservationChecker::new(0.1, 0.01))
            .with_mass(MassConservationChecker::new(1e-6, 1e-4));

        let s0 = state_with(&[
            ("total_energy", 100.0),
            ("momentum_x", 5.0),
            ("momentum_y", 0.0),
            ("momentum_z", 0.0),
            ("total_mass", 1.0),
        ]);
        let s1 = s0.clone();

        assert!(
            suite.all_pass(&[s0, s1]),
            "identical states should all pass"
        );
    }

    #[test]
    fn test_conservation_suite_energy_fails() {
        let suite =
            ConservationSuite::new().with_energy(EnergyConservationChecker::new(0.1, 0.001));

        let s0 = state_with(&[("total_energy", 100.0)]);
        let s1 = state_with(&[("total_energy", 200.0)]);
        let reports = suite.check_trajectory(&[s0, s1]);
        assert_eq!(reports.len(), 1);
        assert!(!reports[0].passed);
    }

    #[test]
    fn test_conservation_suite_empty_passes() {
        let suite = ConservationSuite::new();
        let s = state_with(&[("total_energy", 0.0)]);
        let reports = suite.check_trajectory(&[s]);
        assert!(reports.is_empty());
    }

    #[test]
    fn test_report_has_critical_violations() {
        let checker = EnergyConservationChecker::new(1.0, 1.0);
        let s0 = state_with(&[("total_energy", 100.0)]);
        let s1 = state_with(&[("total_energy", 200.0)]); // 100 > 10× tolerance
        let report = checker.check_pair(&s0, &s1, None);
        assert!(report.has_critical_violations());
    }

    // ── Additional Conservation tests ─────────────────────────────────────────

    #[test]
    fn test_energy_checker_default_tolerances() {
        let checker = EnergyConservationChecker::default_tolerances();
        // 1 mJ absolute, 0.1% relative
        let s0 = state_with(&[("total_energy", 1000.0)]);
        let s1 = state_with(&[("total_energy", 1000.0 + 5e-4)]);
        let report = checker.check_pair(&s0, &s1, None);
        assert!(
            report.passed,
            "small drift should pass with default tolerances"
        );
    }

    #[test]
    fn test_energy_checker_kinetic_plus_potential() {
        let checker = EnergyConservationChecker::new(1.0, 0.01);
        // State without total_energy but with kinetic + potential
        let s0 = state_with(&[("kinetic_energy", 100.0), ("potential_energy", 200.0)]);
        let s1 = state_with(&[("kinetic_energy", 200.0), ("potential_energy", 101.0)]);
        // Change = |301 - 300| = 1.0, at abs_tolerance = 1.0 boundary → check logic
        let report = checker.check_pair(&s0, &s1, None);
        // delta = 1.0, abs_tolerance = 1.0, so NOT > tolerance → pass
        assert!(report.passed, "energy swap within tolerance should pass");
    }

    #[test]
    fn test_energy_checker_trajectory_violations_recorded() {
        let checker = EnergyConservationChecker::new(0.5, 0.01);
        let states = vec![
            state_with(&[("total_energy", 100.0)]),
            state_with(&[("total_energy", 150.0)]), // +50 → violation
            state_with(&[("total_energy", 200.0)]), // +50 → violation
        ];
        let report = checker.check_trajectory(&states);
        assert!(!report.passed);
        assert_eq!(report.states_checked, 2);
        assert_eq!(report.violations.len(), 2);
    }

    #[test]
    fn test_energy_checker_single_state_trajectory_passes() {
        let checker = EnergyConservationChecker::new(1.0, 0.01);
        let states = vec![state_with(&[("total_energy", 100.0)])];
        let report = checker.check_trajectory(&states);
        assert!(report.passed);
        assert_eq!(report.states_checked, 1);
    }

    #[test]
    fn test_momentum_checker_default_tolerances() {
        let checker = MomentumConservationChecker::default_tolerances();
        let s0 = state_with(&[
            ("momentum_x", 5.0),
            ("momentum_y", 0.0),
            ("momentum_z", 0.0),
        ]);
        let s1 = state_with(&[
            ("momentum_x", 5.0),
            ("momentum_y", 1e-8),
            ("momentum_z", 0.0),
        ]);
        let report = checker.check_pair(&s0, &s1, None);
        assert!(
            report.passed,
            "tiny drift with default tolerances should pass"
        );
    }

    #[test]
    fn test_momentum_checker_no_angular() {
        let checker = MomentumConservationChecker {
            abs_tolerance: 0.1,
            rel_tolerance: 0.01,
            check_angular: false,
        };
        let s0 = state_with(&[("momentum_x", 1.0), ("angular_momentum", 100.0)]);
        let s1 = state_with(&[
            ("momentum_x", 1.0),
            ("angular_momentum", 200.0), // big change but angular check disabled
        ]);
        let report = checker.check_pair(&s0, &s1, None);
        assert!(
            report.passed,
            "angular momentum change ignored when check_angular=false"
        );
    }

    #[test]
    fn test_momentum_checker_3d_violation() {
        let checker = MomentumConservationChecker::new(0.001, 0.001);
        let s0 = state_with(&[
            ("momentum_x", 10.0),
            ("momentum_y", 5.0),
            ("momentum_z", 0.0),
        ]);
        let s1 = state_with(&[
            ("momentum_x", 10.0),
            ("momentum_y", 10.0), // +5 → violation
            ("momentum_z", 0.0),
        ]);
        let report = checker.check_pair(&s0, &s1, None);
        assert!(!report.passed, "y-momentum change should violate");
    }

    #[test]
    fn test_mass_checker_default_tolerances() {
        let checker = MassConservationChecker::default_tolerances();
        let s0 = state_with(&[("total_mass", 1.0)]);
        let s1 = state_with(&[("total_mass", 1.0 + 1e-10)]);
        let report = checker.check_pair(&s0, &s1, None);
        assert!(
            report.passed,
            "tiny drift within default tolerances should pass"
        );
    }

    #[test]
    fn test_mass_checker_uses_mass_key() {
        // Falls back to "mass" key when "total_mass" absent
        let checker = MassConservationChecker::new(1e-6, 1e-4);
        let s0 = state_with(&[("mass", 2.0)]);
        let s1 = state_with(&[("mass", 2.0 + 1e-8)]);
        let report = checker.check_pair(&s0, &s1, None);
        assert!(report.passed, "fallback to 'mass' key should work");
    }

    #[test]
    fn test_conservation_report_max_absolute_change_tracked() {
        let checker = EnergyConservationChecker::new(1000.0, 1.0); // very lenient
        let states = vec![
            state_with(&[("total_energy", 100.0)]),
            state_with(&[("total_energy", 150.0)]), // delta=50
            state_with(&[("total_energy", 140.0)]), // delta=10
        ];
        let report = checker.check_trajectory(&states);
        assert!(
            report.max_absolute_change >= 50.0,
            "max change should be tracked"
        );
    }

    #[test]
    fn test_conservation_violation_is_critical_threshold() {
        let checker = EnergyConservationChecker::new(1.0, 1.0);
        let s0 = state_with(&[("total_energy", 100.0)]);
        let s1 = state_with(&[("total_energy", 115.0)]); // delta=15 > 10×1=10 → Critical
        let report = checker.check_pair(&s0, &s1, None);
        assert!(!report.passed);
        assert!(report.violations[0].is_critical());
        assert_eq!(report.violations[0].severity, ViolationSeverity::Critical);
    }

    #[test]
    fn test_conservation_violation_warning_level() {
        let checker = EnergyConservationChecker::new(1.0, 1.0);
        let s0 = state_with(&[("total_energy", 100.0)]);
        let s1 = state_with(&[("total_energy", 105.0)]); // delta=5 > 1, <= 10 → Warning
        let report = checker.check_pair(&s0, &s1, None);
        assert!(!report.passed);
        assert!(!report.violations[0].is_critical());
        assert_eq!(report.violations[0].severity, ViolationSeverity::Warning);
    }

    #[test]
    fn test_conservation_suite_all_pass_with_all_checkers() {
        let suite = ConservationSuite::new()
            .with_energy(EnergyConservationChecker::new(10.0, 0.1))
            .with_momentum(MomentumConservationChecker::new(1.0, 0.1))
            .with_mass(MassConservationChecker::new(0.1, 0.01));

        let state = state_with(&[
            ("total_energy", 500.0),
            ("momentum_x", 10.0),
            ("momentum_y", 5.0),
            ("momentum_z", 0.0),
            ("total_mass", 2.5),
        ]);

        // Identical states → no violations
        let result = suite.all_pass(&[state.clone(), state]);
        assert!(result, "identical states must pass all checkers");
    }

    #[test]
    fn test_conservation_suite_check_trajectory_returns_all_reports() {
        let suite = ConservationSuite::new()
            .with_energy(EnergyConservationChecker::new(1.0, 0.01))
            .with_momentum(MomentumConservationChecker::new(0.01, 0.01))
            .with_mass(MassConservationChecker::new(1e-6, 1e-4));

        let s0 = state_with(&[
            ("total_energy", 100.0),
            ("momentum_x", 5.0),
            ("total_mass", 1.0),
        ]);
        let s1 = s0.clone();

        let reports = suite.check_trajectory(&[s0, s1]);
        // 3 checkers enabled → 3 reports
        assert_eq!(reports.len(), 3, "should produce one report per checker");
        assert!(reports.iter().all(|r| r.passed), "all reports should pass");
    }

    #[test]
    fn test_conservation_report_checker_name() {
        let checker = EnergyConservationChecker::new(1.0, 0.01);
        let s0 = state_with(&[("total_energy", 100.0)]);
        let s1 = s0.clone();
        let report = checker.check_pair(&s0, &s1, None);
        assert_eq!(report.checker_name, "EnergyConservationChecker");
    }

    #[test]
    fn test_mass_checker_trajectory_empty_states() {
        let checker = MassConservationChecker::new(1e-6, 1e-4);
        let report = checker.check_trajectory(&[]);
        assert!(report.passed, "empty trajectory should pass trivially");
        assert_eq!(report.states_checked, 0);
    }

    #[test]
    fn test_momentum_checker_trajectory_clean() {
        let checker = MomentumConservationChecker::new(0.01, 0.001);
        let states: Vec<_> = (0..5)
            .map(|_| {
                state_with(&[
                    ("momentum_x", 1.0),
                    ("momentum_y", 0.0),
                    ("momentum_z", 0.0),
                ])
            })
            .collect();
        let report = checker.check_trajectory(&states);
        assert!(report.passed, "constant momentum trajectory must pass");
    }
}

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

/// A symmetry type in the Noether theorem framework.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum PhysicalSymmetry {
    /// Time translation symmetry → energy is conserved.
    TimeTranslation,
    /// Spatial translation symmetry → linear momentum is conserved.
    SpatialTranslation,
    /// Rotational symmetry → angular momentum is conserved.
    Rotation,
}

impl PhysicalSymmetry {
    /// Human-readable name of the symmetry.
    pub fn name(self) -> &'static str {
        match self {
            Self::TimeTranslation => "Time Translation",
            Self::SpatialTranslation => "Spatial Translation",
            Self::Rotation => "Rotational",
        }
    }

    /// Name of the corresponding conserved quantity.
    pub fn conserved_quantity(self) -> &'static str {
        match self {
            Self::TimeTranslation => "energy",
            Self::SpatialTranslation => "linear momentum",
            Self::Rotation => "angular momentum",
        }
    }
}

/// Result of a Noether symmetry check.
#[derive(Debug, Clone)]
pub struct NoetherCheckResult {
    /// The symmetry that was checked.
    pub symmetry: PhysicalSymmetry,
    /// Whether the corresponding conserved quantity is indeed conserved.
    pub conserved: bool,
    /// The underlying conservation report.
    pub report: ConservationReport,
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

// ──────────────────────────────────────────────────────────────────────────────
// Tests — Entropy, Angular Momentum, Noether
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod entropy_angular_noether_tests {
    use super::super::PhysState;
    use super::*;

    fn state(pairs: &[(&str, f64)]) -> PhysState {
        let mut s = PhysState::new();
        for &(k, v) in pairs {
            s.set(k, v);
        }
        s
    }

    // ── EntropyConservationChecker ────────────────────────────────────────────

    #[test]
    fn test_entropy_non_decrease_pass() {
        let checker = EntropyConservationChecker::new(1e-6);
        let s0 = state(&[("entropy", 100.0)]);
        let s1 = state(&[("entropy", 105.0)]); // entropy increases → OK
        let report = checker.check_pair(&s0, &s1, None);
        assert!(report.passed, "increasing entropy must pass");
    }

    #[test]
    fn test_entropy_decrease_violation() {
        let checker = EntropyConservationChecker::new(1e-6);
        let s0 = state(&[("entropy", 100.0)]);
        let s1 = state(&[("entropy", 90.0)]); // entropy decreases → violation
        let report = checker.check_pair(&s0, &s1, None);
        assert!(!report.passed, "decreasing entropy must violate");
        assert!(!report.violations.is_empty());
    }

    #[test]
    fn test_entropy_monotonicity_trajectory() {
        let checker = EntropyConservationChecker::new(1e-6);
        let states = vec![
            state(&[("entropy", 10.0)]),
            state(&[("entropy", 10.5)]),
            state(&[("entropy", 11.0)]),
            state(&[("entropy", 11.0)]), // plateau is fine (dS >= 0)
        ];
        let report = checker.check_trajectory(&states);
        assert!(
            report.passed,
            "monotone non-decreasing trajectory must pass"
        );
    }

    #[test]
    fn test_entropy_trajectory_violation_at_step() {
        let checker = EntropyConservationChecker::new(1e-6);
        let states = vec![
            state(&[("entropy", 10.0)]),
            state(&[("entropy", 12.0)]),
            state(&[("entropy", 8.0)]), // decrease → violation at step 1
        ];
        let report = checker.check_trajectory(&states);
        assert!(!report.passed);
        assert!(
            report.violations.iter().any(|v| v.step_index == Some(1)),
            "violation should be at step 1"
        );
    }

    #[test]
    fn test_clausius_inequality_satisfied() {
        let checker = EntropyConservationChecker::new(1e-6);
        // dS = 10 J/K, Q = 50 J, T = 300 K → Q/T = 0.167 J/K < 10 → satisfied
        let s0 = state(&[("entropy", 100.0)]);
        let s1 = state(&[
            ("entropy", 110.0),
            ("heat_flow", 50.0),
            ("temperature", 300.0),
        ]);
        let report = checker.check_pair(&s0, &s1, None);
        assert!(report.passed, "Clausius inequality satisfied");
    }

    // ── AngularMomentumChecker ────────────────────────────────────────────────

    #[test]
    fn test_angular_momentum_conservation_pass() {
        let checker = AngularMomentumChecker::new(1e-6, 1e-3).without_torque_check();
        let s0 = state(&[
            ("angular_momentum_x", 5.0),
            ("angular_momentum_y", 3.0),
            ("angular_momentum_z", 1.0),
        ]);
        let s1 = s0.clone();
        let report = checker.check_pair(&s0, &s1, None);
        assert!(
            report.passed,
            "identical states must conserve angular momentum"
        );
    }

    #[test]
    fn test_angular_momentum_violation() {
        let checker = AngularMomentumChecker::new(1e-4, 1e-3).without_torque_check();
        let s0 = state(&[
            ("angular_momentum_x", 5.0),
            ("angular_momentum_y", 0.0),
            ("angular_momentum_z", 0.0),
        ]);
        let s1 = state(&[
            ("angular_momentum_x", 5.0),
            ("angular_momentum_y", 2.0), // change in y component
            ("angular_momentum_z", 0.0),
        ]);
        let report = checker.check_pair(&s0, &s1, None);
        assert!(!report.passed, "y-component change must violate");
    }

    #[test]
    fn test_angular_momentum_central_force_conservation() {
        // In a central force orbit, angular momentum is conserved
        let checker = AngularMomentumChecker::default_tolerances().without_torque_check();
        let l_z = 4.0; // conserved z-component (orbit in xy-plane)
        let states: Vec<PhysState> = (0..5)
            .map(|_| {
                state(&[
                    ("angular_momentum_x", 0.0),
                    ("angular_momentum_y", 0.0),
                    ("angular_momentum_z", l_z),
                ])
            })
            .collect();
        let report = checker.check_trajectory(&states);
        assert!(
            report.passed,
            "central force orbit conserves angular momentum"
        );
    }

    // ── NoetherSymmetryValidator ──────────────────────────────────────────────

    #[test]
    fn test_noether_time_translation_to_energy() {
        let validator = NoetherSymmetryValidator::new(vec![PhysicalSymmetry::TimeTranslation], 1.0);
        let states = vec![
            state(&[("total_energy", 100.0)]),
            state(&[("total_energy", 100.0)]),
            state(&[("total_energy", 100.0)]),
        ];
        let results = validator.validate(&states);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].symmetry, PhysicalSymmetry::TimeTranslation);
        assert!(
            results[0].conserved,
            "constant energy should confirm time translation symmetry"
        );
    }

    #[test]
    fn test_noether_spatial_translation_to_momentum() {
        let validator =
            NoetherSymmetryValidator::new(vec![PhysicalSymmetry::SpatialTranslation], 1e-4);
        let s = state(&[
            ("momentum_x", 5.0),
            ("momentum_y", 0.0),
            ("momentum_z", 0.0),
        ]);
        let states = vec![s.clone(), s.clone(), s];
        let results = validator.validate(&states);
        assert!(
            results[0].conserved,
            "constant momentum → spatial translation symmetry"
        );
    }

    #[test]
    fn test_noether_rotation_to_angular_momentum() {
        let validator = NoetherSymmetryValidator::new(vec![PhysicalSymmetry::Rotation], 1e-4);
        let s = state(&[
            ("angular_momentum_x", 0.0),
            ("angular_momentum_y", 0.0),
            ("angular_momentum_z", 3.0),
        ]);
        let states = vec![s.clone(), s.clone(), s];
        let results = validator.validate(&states);
        assert!(
            results[0].conserved,
            "constant angular momentum → rotational symmetry"
        );
    }

    #[test]
    fn test_noether_all_three_symmetries() {
        let validator = NoetherSymmetryValidator::new(
            vec![
                PhysicalSymmetry::TimeTranslation,
                PhysicalSymmetry::SpatialTranslation,
                PhysicalSymmetry::Rotation,
            ],
            1e-4,
        );
        let s = state(&[
            ("total_energy", 200.0),
            ("momentum_x", 1.0),
            ("momentum_y", 0.0),
            ("momentum_z", 0.0),
            ("angular_momentum_x", 0.0),
            ("angular_momentum_y", 0.0),
            ("angular_momentum_z", 5.0),
        ]);
        let states = vec![s.clone(), s.clone()];
        assert!(
            validator.all_conserved(&states),
            "all three Noether quantities conserved"
        );
    }

    #[test]
    fn test_noether_conserved_quantity_names() {
        assert_eq!(
            PhysicalSymmetry::TimeTranslation.conserved_quantity(),
            "energy"
        );
        assert_eq!(
            PhysicalSymmetry::SpatialTranslation.conserved_quantity(),
            "linear momentum"
        );
        assert_eq!(
            PhysicalSymmetry::Rotation.conserved_quantity(),
            "angular momentum"
        );
    }

    #[test]
    fn test_noether_symmetry_names() {
        assert_eq!(PhysicalSymmetry::TimeTranslation.name(), "Time Translation");
        assert_eq!(
            PhysicalSymmetry::SpatialTranslation.name(),
            "Spatial Translation"
        );
        assert_eq!(PhysicalSymmetry::Rotation.name(), "Rotational");
    }
}
