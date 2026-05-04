//! Physics plausibility scoring using pure arithmetic (no external physics crate).
//!
//! Each [`PhysicsDomain`] encodes a different governing equation and produces a
//! [`PlausibilityScore`] in **[0.0, 1.0]** for a map of named scalar properties.
//!
//! The four domains implemented are:
//! - **ThermalDiffusion** — Fourier number `Fo = α·t/L²`
//! - **FluidFlow** — Reynolds number `Re = v·L/ν`
//! - **StructuralMechanics** — Hooke's law consistency `ε_theory = σ/E`
//! - **Electromagnetic** — Ohm's law `V = I·R`
//!
//! All physics is pure arithmetic; no `oxirs-physics` dependency is needed.

use std::collections::HashMap;

// ─── Public types ─────────────────────────────────────────────────────────────

/// Describes which physical governing equation to use for plausibility checking.
#[derive(Debug, Clone)]
pub enum PhysicsDomain {
    /// Fourier number: `Fo = α·t / L²`.
    ///
    /// Required properties: `"time_s"`, `"length_m"`.
    /// The thermal diffusivity `α` (m²/s) is stored in the variant.
    ThermalDiffusion {
        /// Thermal diffusivity α in m²/s (e.g. steel ≈ 1.2e-5, aluminium ≈ 8.4e-5).
        thermal_diffusivity: f64,
    },

    /// Reynolds number: `Re = v·L / ν`.
    ///
    /// Required properties: `"velocity_ms"`, `"length_m"`.
    /// The kinematic viscosity `ν` (m²/s) and expected regime are stored in the variant.
    FluidFlow {
        /// Kinematic viscosity ν in m²/s (e.g. water at 20°C ≈ 1.0e-6).
        kinematic_viscosity: f64,
        /// Which flow regime is expected for the entity being scored.
        expected_regime: FlowRegime,
    },

    /// Hooke's law consistency: `ε_theory = σ / E` vs measured `ε`.
    ///
    /// Required properties: `"stress_pa"`, `"strain"`.
    StructuralMechanics {
        /// Young's modulus E in Pa (e.g. structural steel ≈ 2.0e11).
        youngs_modulus_pa: f64,
        /// Yield stress in Pa; exceeding it triggers a 0.1× penalty.
        yield_stress_pa: f64,
    },

    /// Ohm's law: `V = I·R`.
    ///
    /// Required properties: `"voltage_v"`, `"current_a"`, `"resistance_ohm"`.
    Electromagnetic,
}

/// Expected hydrodynamic flow regime for a [`PhysicsDomain::FluidFlow`] entity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlowRegime {
    /// Re < 2 300 — viscous, ordered flow.
    Laminar,
    /// Re > 4 000 — inertia-dominated, chaotic flow.
    Turbulent,
    /// 2 300 ≤ Re ≤ 4 000 — mixed / unstable.
    Transitional,
}

/// Result of a physics plausibility check.
#[derive(Debug, Clone)]
pub struct PlausibilityScore {
    /// Plausibility in **[0.0, 1.0]**: 1.0 = fully plausible, 0.0 = implausible.
    pub score: f64,
    /// Human-readable explanation of the score.
    pub reason: String,
    /// The dimensionless parameter computed (Fo, Re, σ/E ratio, V/IR ratio),
    /// or `None` when required properties were missing.
    pub dimensionless_param: Option<f64>,
}

// ─── PhysicsContext ───────────────────────────────────────────────────────────

/// Wraps a [`PhysicsDomain`] and computes plausibility scores from property maps.
pub struct PhysicsContext {
    /// The governing physical domain.
    pub domain: PhysicsDomain,
}

impl PhysicsContext {
    /// Create a new context for the given domain.
    pub fn new(domain: PhysicsDomain) -> Self {
        Self { domain }
    }

    /// Compute a physics plausibility score from a map of named scalar properties.
    ///
    /// Returns 0.5 with an explanation when required properties are absent, so that
    /// missing data is treated as *neutral* rather than implausible.
    pub fn plausibility_score(&self, properties: &HashMap<String, f64>) -> PlausibilityScore {
        match &self.domain {
            PhysicsDomain::ThermalDiffusion {
                thermal_diffusivity,
            } => score_thermal(*thermal_diffusivity, properties),

            PhysicsDomain::FluidFlow {
                kinematic_viscosity,
                expected_regime,
            } => score_fluid(*kinematic_viscosity, *expected_regime, properties),

            PhysicsDomain::StructuralMechanics {
                youngs_modulus_pa,
                yield_stress_pa,
            } => score_structural(*youngs_modulus_pa, *yield_stress_pa, properties),

            PhysicsDomain::Electromagnetic => score_electromagnetic(properties),
        }
    }
}

// ─── Domain scorers ───────────────────────────────────────────────────────────

/// Score thermal plausibility via the Fourier number `Fo = α·t/L²`.
///
/// Scoring rules:
/// * `0.001 ≤ Fo ≤ 1000` → `score = (1 − |log10(Fo)| / 3).clamp(0, 1)`
/// * `Fo` outside that range → decays linearly toward 0 (already implied by clamp)
/// * Missing required properties → score = 0.5
fn score_thermal(alpha: f64, props: &HashMap<String, f64>) -> PlausibilityScore {
    let (t, l) = match (props.get("time_s"), props.get("length_m")) {
        (Some(&t), Some(&l)) => (t, l),
        _ => {
            return PlausibilityScore {
                score: 0.5,
                reason:
                    "Missing required properties 'time_s' or 'length_m'; returning neutral score."
                        .into(),
                dimensionless_param: None,
            };
        }
    };

    if l <= 0.0 || t < 0.0 {
        return PlausibilityScore {
            score: 0.0,
            reason: format!("Non-physical inputs: time_s={t:.3e}, length_m={l:.3e}"),
            dimensionless_param: None,
        };
    }

    let fo = alpha * t / (l * l);
    if fo <= 0.0 {
        return PlausibilityScore {
            score: 0.0,
            reason: format!(
                "Fourier number Fo={fo:.3e} is non-positive (α={alpha:.3e}, t={t:.3e}, L={l:.3e})"
            ),
            dimensionless_param: Some(fo),
        };
    }

    // log10 of Fo; values in [0.001, 1000] → log10 ∈ [−3, 3]
    let log_fo = fo.log10();
    // score peaks at Fo=1 (log=0) and decays to 0 at the boundary ±3
    let score = (1.0 - log_fo.abs() / 3.0).clamp(0.0, 1.0);

    PlausibilityScore {
        score,
        reason: format!("Fourier number Fo={fo:.4e} → score={score:.4}"),
        dimensionless_param: Some(fo),
    }
}

/// Score fluid plausibility via the Reynolds number `Re = v·L/ν`.
///
/// Scoring rules:
/// * **Laminar** (expected Re < 2 300):
///   - score = 1.0 if Re ≤ 2 300
///   - linear decay from 1.0 at Re = 2 300 to 0.0 at Re = 10 000
/// * **Turbulent** (expected Re > 4 000):
///   - score = 1.0 if Re ≥ 4 000
///   - linear decay from 1.0 at Re = 4 000 to 0.0 at Re = 100
/// * **Transitional** (expected 2 300 ≤ Re ≤ 4 000):
///   - score = 1.0 inside [2 300, 4 000]
///   - decays linearly to 0.0 below Re = 1 000 or above Re = 10 000
/// * Missing properties → score = 0.5
fn score_fluid(nu: f64, expected: FlowRegime, props: &HashMap<String, f64>) -> PlausibilityScore {
    let (v, l) = match (props.get("velocity_ms"), props.get("length_m")) {
        (Some(&v), Some(&l)) => (v, l),
        _ => {
            return PlausibilityScore {
                score: 0.5,
                reason: "Missing required properties 'velocity_ms' or 'length_m'; returning neutral score.".into(),
                dimensionless_param: None,
            };
        }
    };

    if nu <= 0.0 || l <= 0.0 || v < 0.0 {
        return PlausibilityScore {
            score: 0.0,
            reason: format!(
                "Non-physical inputs: velocity_ms={v:.3e}, length_m={l:.3e}, nu={nu:.3e}"
            ),
            dimensionless_param: None,
        };
    }

    let re = v * l / nu;

    let score = match expected {
        FlowRegime::Laminar => {
            // Perfect below 2300; linear decay to 0 at Re = 10 000
            if re <= 2300.0 {
                1.0
            } else {
                (1.0 - (re - 2300.0) / (10_000.0 - 2300.0)).clamp(0.0, 1.0)
            }
        }
        FlowRegime::Turbulent => {
            // Perfect above 4000; linear decay to 0 at Re = 100
            if re >= 4000.0 {
                1.0
            } else {
                ((re - 100.0) / (4000.0 - 100.0)).clamp(0.0, 1.0)
            }
        }
        FlowRegime::Transitional => {
            // Perfect in [2300, 4000]; linear flanks reaching 0 at Re = 1000 and Re = 10 000
            if (2300.0..=4000.0).contains(&re) {
                1.0
            } else if re < 2300.0 {
                ((re - 1000.0) / (2300.0 - 1000.0)).clamp(0.0, 1.0)
            } else {
                (1.0 - (re - 4000.0) / (10_000.0 - 4000.0)).clamp(0.0, 1.0)
            }
        }
    };

    let regime_str = match expected {
        FlowRegime::Laminar => "laminar",
        FlowRegime::Turbulent => "turbulent",
        FlowRegime::Transitional => "transitional",
    };
    PlausibilityScore {
        score,
        reason: format!("Reynolds number Re={re:.2} (expected {regime_str}) → score={score:.4}"),
        dimensionless_param: Some(re),
    }
}

/// Score structural plausibility via Hooke's law: `ε_theory = σ/E` vs `ε_measured`.
///
/// Scoring rules:
/// * `score = (1 − |ε_meas − ε_theory| / ε_theory).max(0.0)` when ε_theory > 0
/// * If `σ > yield_stress` → `score *= 0.1`
/// * Missing properties → score = 0.5
fn score_structural(
    youngs_pa: f64,
    yield_pa: f64,
    props: &HashMap<String, f64>,
) -> PlausibilityScore {
    let (sigma, eps_meas) = match (props.get("stress_pa"), props.get("strain")) {
        (Some(&s), Some(&e)) => (s, e),
        _ => {
            return PlausibilityScore {
                score: 0.5,
                reason:
                    "Missing required properties 'stress_pa' or 'strain'; returning neutral score."
                        .into(),
                dimensionless_param: None,
            };
        }
    };

    if youngs_pa <= 0.0 {
        return PlausibilityScore {
            score: 0.0,
            reason: format!("Non-physical Young's modulus: {youngs_pa:.3e} Pa"),
            dimensionless_param: None,
        };
    }

    let eps_theory = sigma / youngs_pa;
    let ratio = sigma / youngs_pa; // dimensionless σ/E

    let base_score = if eps_theory.abs() < 1e-30 {
        // Degenerate case: zero stress → treat as perfect Hooke consistency
        if eps_meas.abs() < 1e-15 {
            1.0
        } else {
            0.0
        }
    } else {
        let rel_err = (eps_meas - eps_theory).abs() / eps_theory.abs();
        (1.0 - rel_err).max(0.0)
    };

    let (score, yield_note) = if sigma.abs() > yield_pa {
        (base_score * 0.1, " (yield exceeded → ×0.1 penalty)")
    } else {
        (base_score, "")
    };

    PlausibilityScore {
        score,
        reason: format!(
            "Hooke: σ={sigma:.3e} Pa, E={youngs_pa:.3e} Pa → ε_theory={eps_theory:.4e}, \
             ε_meas={eps_meas:.4e}, score={score:.4}{yield_note}"
        ),
        dimensionless_param: Some(ratio),
    }
}

/// Score electromagnetic plausibility via Ohm's law: `V_theory = I·R`.
///
/// Scoring rules:
/// * `score = (1 − |V − V_theory| / |V|.max(1e-9)).clamp(0, 1)`
/// * Missing properties → score = 0.5
fn score_electromagnetic(props: &HashMap<String, f64>) -> PlausibilityScore {
    let (v, i, r) = match (
        props.get("voltage_v"),
        props.get("current_a"),
        props.get("resistance_ohm"),
    ) {
        (Some(&v), Some(&i), Some(&r)) => (v, i, r),
        _ => {
            return PlausibilityScore {
                score: 0.5,
                reason: "Missing required properties 'voltage_v', 'current_a', or \
                         'resistance_ohm'; returning neutral score."
                    .into(),
                dimensionless_param: None,
            };
        }
    };

    if r < 0.0 {
        return PlausibilityScore {
            score: 0.0,
            reason: format!("Non-physical negative resistance: {r:.3e} Ω"),
            dimensionless_param: None,
        };
    }

    let v_theory = i * r;
    let denominator = v.abs().max(1e-9);
    let ratio = v_theory / denominator;
    let score = (1.0 - (v - v_theory).abs() / denominator).clamp(0.0, 1.0);

    PlausibilityScore {
        score,
        reason: format!("Ohm: V={v:.4} V, I·R={v_theory:.4} V → score={score:.4}"),
        dimensionless_param: Some(ratio),
    }
}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn thermal_ctx(alpha: f64) -> PhysicsContext {
        PhysicsContext::new(PhysicsDomain::ThermalDiffusion {
            thermal_diffusivity: alpha,
        })
    }

    fn props(pairs: &[(&str, f64)]) -> HashMap<String, f64> {
        pairs.iter().map(|(k, v)| ((*k).to_string(), *v)).collect()
    }

    #[test]
    fn test_thermal_fo_one_perfect() {
        // Fo = α·t/L² = 1e-5·1.0/1e-5 = 1.0 → log10(1) = 0 → score = 1
        let ctx = thermal_ctx(1e-5);
        let p = ctx.plausibility_score(&props(&[("time_s", 1.0), ("length_m", (1e-5_f64).sqrt())]));
        assert!(p.score > 0.99, "expected ~1.0, got {}", p.score);
    }

    #[test]
    fn test_thermal_missing_props_neutral() {
        let ctx = thermal_ctx(1e-5);
        let p = ctx.plausibility_score(&props(&[]));
        assert!((p.score - 0.5).abs() < 1e-10);
        assert!(p.dimensionless_param.is_none());
    }

    #[test]
    fn test_fluid_laminar_correct_scores_high() {
        let ctx = PhysicsContext::new(PhysicsDomain::FluidFlow {
            kinematic_viscosity: 1e-6,
            expected_regime: FlowRegime::Laminar,
        });
        // Re = 0.01 * 0.1 / 1e-6 = 1000 — laminar
        let p = ctx.plausibility_score(&props(&[("velocity_ms", 0.01), ("length_m", 0.1)]));
        assert!(
            p.score > 0.9,
            "expected high laminar score, got {}",
            p.score
        );
    }

    #[test]
    fn test_fluid_turbulent_missing_props_neutral() {
        let ctx = PhysicsContext::new(PhysicsDomain::FluidFlow {
            kinematic_viscosity: 1e-6,
            expected_regime: FlowRegime::Turbulent,
        });
        let p = ctx.plausibility_score(&props(&[]));
        assert!((p.score - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_structural_yield_exceeded_low() {
        let ctx = PhysicsContext::new(PhysicsDomain::StructuralMechanics {
            youngs_modulus_pa: 2e11,
            yield_stress_pa: 2.5e8,
        });
        // σ well above yield, strain exactly matches Hooke → base=1.0 → ×0.1
        let sigma = 5e8_f64;
        let eps = sigma / 2e11;
        let p = ctx.plausibility_score(&props(&[("stress_pa", sigma), ("strain", eps)]));
        assert!(
            p.score < 0.2,
            "expected low score due to yield, got {}",
            p.score
        );
    }

    #[test]
    fn test_electromagnetic_ohm_consistent() {
        let ctx = PhysicsContext::new(PhysicsDomain::Electromagnetic);
        // V = I·R exactly
        let p = ctx.plausibility_score(&props(&[
            ("voltage_v", 12.0),
            ("current_a", 2.0),
            ("resistance_ohm", 6.0),
        ]));
        assert!(
            p.score > 0.99,
            "expected ~1.0 for exact Ohm, got {}",
            p.score
        );
    }

    #[test]
    fn test_electromagnetic_missing_props_neutral() {
        let ctx = PhysicsContext::new(PhysicsDomain::Electromagnetic);
        let p = ctx.plausibility_score(&props(&[("voltage_v", 10.0)]));
        assert!((p.score - 0.5).abs() < 1e-10);
    }
}
