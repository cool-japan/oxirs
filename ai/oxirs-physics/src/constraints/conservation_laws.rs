//! Conservation Law Validation
//!
//! Validates simulation results against fundamental physics conservation laws

use crate::simulation::result_injection::StateVector;

/// Conservation Law Types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConservationLaw {
    /// Energy conservation: dE/dt = 0 (isolated system)
    Energy,

    /// Mass conservation: dm/dt = 0
    Mass,

    /// Momentum conservation: dp/dt = F_external
    Momentum,

    /// Angular momentum: dL/dt = τ_external
    AngularMomentum,

    /// Charge conservation
    Charge,
}

impl ConservationLaw {
    /// Get the law name
    pub fn name(&self) -> &'static str {
        match self {
            ConservationLaw::Energy => "Energy Conservation",
            ConservationLaw::Mass => "Mass Conservation",
            ConservationLaw::Momentum => "Momentum Conservation",
            ConservationLaw::AngularMomentum => "Angular Momentum Conservation",
            ConservationLaw::Charge => "Charge Conservation",
        }
    }
}

/// Conservation Checker
pub struct ConservationChecker {
    laws: Vec<ConservationLaw>,
    tolerance: f64,
}

impl ConservationChecker {
    /// Create a new conservation checker
    pub fn new(tolerance: f64) -> Self {
        Self {
            laws: vec![ConservationLaw::Energy, ConservationLaw::Mass],
            tolerance,
        }
    }

    /// Add a conservation law to check
    pub fn add_law(&mut self, law: ConservationLaw) {
        if !self.laws.contains(&law) {
            self.laws.push(law);
        }
    }

    /// Check conservation laws on trajectory
    pub fn check(&self, trajectory: &[StateVector]) -> Vec<ViolationReport> {
        self.laws
            .iter()
            .filter_map(|law| self.check_law(*law, trajectory))
            .collect()
    }

    /// Check a specific conservation law
    ///
    /// Walks every recorded state after the first (not just the final
    /// state), tracking the worst deviation from the initial value seen
    /// anywhere in the trajectory. This ensures a mid-trajectory
    /// conservation violation (e.g. a value that spikes and then returns
    /// close to its initial value by the final recorded step) is still
    /// detected, matching the stricter per-step checking done by
    /// `conservation::checkers_impl`.
    fn check_law(
        &self,
        law: ConservationLaw,
        trajectory: &[StateVector],
    ) -> Option<ViolationReport> {
        if trajectory.len() < 2 {
            return None;
        }

        let quantity_name = match law {
            ConservationLaw::Energy => "energy",
            ConservationLaw::Mass => "mass",
            ConservationLaw::Momentum => "momentum",
            ConservationLaw::AngularMomentum => "angular_momentum",
            ConservationLaw::Charge => "charge",
        };

        let initial_value = trajectory
            .first()?
            .state
            .get(quantity_name)
            .copied()
            .unwrap_or(0.0);

        // Track the state with the worst (largest relative) deviation from
        // the initial value across the *entire* trajectory, not just the
        // final state.
        let mut worst_value = initial_value;
        let mut worst_change = 0.0_f64;
        let mut worst_relative_change = 0.0_f64;

        for state in trajectory.iter().skip(1) {
            let value = state.state.get(quantity_name).copied().unwrap_or(0.0);
            let change = (value - initial_value).abs();
            let relative_change = if initial_value.abs() > 1e-10 {
                change / initial_value.abs()
            } else {
                change
            };

            if relative_change > worst_relative_change {
                worst_value = value;
                worst_change = change;
                worst_relative_change = relative_change;
            }
        }

        if worst_relative_change > self.tolerance {
            Some(ViolationReport {
                law: law.name().to_string(),
                initial_value,
                final_value: worst_value,
                change: worst_change,
                relative_change: worst_relative_change,
                tolerance: self.tolerance,
            })
        } else {
            None
        }
    }
}

impl Default for ConservationChecker {
    fn default() -> Self {
        Self::new(0.01) // 1% tolerance
    }
}

/// Violation Report
#[derive(Debug, Clone)]
pub struct ViolationReport {
    pub law: String,
    pub initial_value: f64,
    pub final_value: f64,
    pub change: f64,
    pub relative_change: f64,
    pub tolerance: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conservation_checker() {
        let checker = ConservationChecker::new(0.01);

        let mut trajectory = Vec::new();

        for i in 0..10 {
            let mut state = std::collections::HashMap::new();
            state.insert("energy".to_string(), 100.0); // Constant energy
            state.insert("mass".to_string(), 50.0); // Constant mass

            trajectory.push(StateVector {
                time: i as f64,
                state,
            });
        }

        let violations = checker.check(&trajectory);
        assert!(violations.is_empty());
    }

    #[test]
    fn test_conservation_violation() {
        let checker = ConservationChecker::new(0.01);

        let mut trajectory = Vec::new();

        for i in 0..10 {
            let mut state = std::collections::HashMap::new();
            state.insert("energy".to_string(), 100.0 + i as f64 * 10.0); // Increasing energy

            trajectory.push(StateVector {
                time: i as f64,
                state,
            });
        }

        let violations = checker.check(&trajectory);
        assert!(!violations.is_empty());
        assert_eq!(violations[0].law, "Energy Conservation");
    }

    /// Regression test for the P1 finding: `check_law` used to compare only
    /// `trajectory.first()` and `trajectory.last()`, so a violation that
    /// spikes mid-trajectory and then returns close to the initial value by
    /// the final recorded step was invisible. This trajectory starts and
    /// ends near 100.0 (well within the 1% tolerance end-to-end) but spikes
    /// to 1000.0 in the middle, which must be caught.
    #[test]
    fn regression_mid_trajectory_spike_is_detected() {
        let checker = ConservationChecker::new(0.01);

        let mut trajectory = Vec::new();
        let energies = [100.0, 100.5, 1000.0, 100.5, 100.2, 100.0];
        for (i, &e) in energies.iter().enumerate() {
            let mut state = std::collections::HashMap::new();
            state.insert("energy".to_string(), e);
            trajectory.push(StateVector {
                time: i as f64,
                state,
            });
        }

        // Sanity check: first and last values alone are within tolerance,
        // so the old endpoints-only comparison would have reported no
        // violation at all.
        let first = trajectory.first().expect("non-empty").state["energy"];
        let last = trajectory.last().expect("non-empty").state["energy"];
        assert!((last - first).abs() / first.abs() < 0.01);

        let violations = checker.check(&trajectory);
        assert!(
            !violations.is_empty(),
            "mid-trajectory spike to 1000.0 must be detected even though the endpoints match"
        );
        let energy_violation = violations
            .iter()
            .find(|v| v.law == "Energy Conservation")
            .expect("expected an energy conservation violation");
        assert!((energy_violation.final_value - 1000.0).abs() < 1e-9);
    }

    /// Regression test: a trajectory that never deviates beyond tolerance
    /// at any intermediate step (not just the endpoints) must still report
    /// no violations.
    #[test]
    fn regression_no_false_positive_within_tolerance_throughout() {
        let checker = ConservationChecker::new(0.05);

        let mut trajectory = Vec::new();
        // Small oscillation within 5% band around 100.0 at every step.
        let energies = [100.0, 102.0, 98.0, 103.0, 99.0, 101.0];
        for (i, &e) in energies.iter().enumerate() {
            let mut state = std::collections::HashMap::new();
            state.insert("energy".to_string(), e);
            trajectory.push(StateVector {
                time: i as f64,
                state,
            });
        }

        let violations = checker.check(&trajectory);
        assert!(
            violations.is_empty(),
            "small in-tolerance oscillation must not be flagged: {violations:?}"
        );
    }
}
