//! Types and data structures for conservation law checkers.
//!
//! Contains all structs, enums, and their core implementations
//! used by the enhanced conservation law checker infrastructure.

use serde::{Deserialize, Serialize};

// ──────────────────────────────────────────────────────────────────────────────
// ViolationSeverity
// ──────────────────────────────────────────────────────────────────────────────

/// Severity level of a conservation violation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ViolationSeverity {
    /// Within 10× the configured tolerance.
    Warning,
    /// Outside 10× the configured tolerance.
    Critical,
}

// ──────────────────────────────────────────────────────────────────────────────
// ConservationViolationDetail
// ──────────────────────────────────────────────────────────────────────────────

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
    pub(crate) fn new(
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
    pub(crate) fn new(checker_name: impl Into<String>) -> Self {
        Self {
            checker_name: checker_name.into(),
            passed: true,
            violations: Vec::new(),
            states_checked: 0,
            max_absolute_change: 0.0,
        }
    }

    pub(crate) fn add_violation(&mut self, detail: ConservationViolationDetail) {
        self.max_absolute_change = self.max_absolute_change.max(detail.absolute_change.abs());
        self.violations.push(detail);
        self.passed = false;
    }

    pub(crate) fn track_change(&mut self, change: f64) {
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
// PhysicalSymmetry
// ──────────────────────────────────────────────────────────────────────────────

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

// ──────────────────────────────────────────────────────────────────────────────
// NoetherCheckResult
// ──────────────────────────────────────────────────────────────────────────────

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
