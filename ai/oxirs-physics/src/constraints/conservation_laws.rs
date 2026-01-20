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

        // Extract quantity from first and last state
        let first_value = trajectory
            .first()?
            .state
            .get(quantity_name)
            .copied()
            .unwrap_or(0.0);

        let last_value = trajectory
            .last()?
            .state
            .get(quantity_name)
            .copied()
            .unwrap_or(0.0);

        let change = (last_value - first_value).abs();
        let relative_change = if first_value.abs() > 1e-10 {
            change / first_value.abs()
        } else {
            change
        };

        if relative_change > self.tolerance {
            Some(ViolationReport {
                law: law.name().to_string(),
                initial_value: first_value,
                final_value: last_value,
                change,
                relative_change,
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
}
