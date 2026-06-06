#[cfg(test)]
mod tests {
    use crate::conservation::checkers_impl::{
        ConservationSuite, EnergyConservationChecker, MassConservationChecker,
        MomentumConservationChecker,
    };
    use crate::conservation::checkers_types::ViolationSeverity;
    use crate::conservation::PhysState;

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

#[cfg(test)]
mod entropy_angular_noether_tests {
    use crate::conservation::checkers_types::PhysicalSymmetry;
    use crate::conservation::checkers_validator::{
        AngularMomentumChecker, EntropyConservationChecker, NoetherSymmetryValidator,
    };
    use crate::conservation::PhysState;

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
