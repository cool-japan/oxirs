//! Basic computational fluid dynamics calculations.
//!
//! Provides pipe flow analysis, Bernoulli equation, heat transfer coefficients,
//! and drag force calculations for common fluid dynamics problems.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Physical constants
// ---------------------------------------------------------------------------

/// Water density at 20 °C in kg/m³
pub const WATER_DENSITY_KG_M3: f64 = 998.2;
/// Air density at 20 °C, sea level, in kg/m³
pub const AIR_DENSITY_KG_M3: f64 = 1.204;
/// Dynamic viscosity of water at 20 °C in Pa·s
pub const WATER_VISCOSITY_PA_S: f64 = 1.002e-3;
/// Dynamic viscosity of air at 20 °C in Pa·s
pub const AIR_VISCOSITY_PA_S: f64 = 1.825e-5;
/// Standard gravity in m/s²
pub const GRAVITY_M_S2: f64 = 9.81;

// ---------------------------------------------------------------------------
// FluidProperties
// ---------------------------------------------------------------------------

/// Thermophysical properties of a fluid.
#[derive(Debug, Clone)]
pub struct FluidProperties {
    /// Human-readable name
    pub name: String,
    /// Density in kg/m³
    pub density: f64,
    /// Dynamic viscosity in Pa·s
    pub dynamic_viscosity: f64,
    /// Kinematic viscosity in m²/s (= dynamic / density)
    pub kinematic_viscosity: f64,
    /// Specific heat capacity in J/(kg·K)
    pub specific_heat: f64,
    /// Thermal conductivity in W/(m·K)
    pub thermal_conductivity: f64,
}

impl FluidProperties {
    /// Water properties at 20 °C.
    pub fn water() -> Self {
        let density = WATER_DENSITY_KG_M3;
        let dynamic_viscosity = WATER_VISCOSITY_PA_S;
        Self {
            name: "Water".to_string(),
            density,
            dynamic_viscosity,
            kinematic_viscosity: dynamic_viscosity / density,
            specific_heat: 4182.0,
            thermal_conductivity: 0.598,
        }
    }

    /// Air properties at 20 °C and sea level.
    pub fn air() -> Self {
        let density = AIR_DENSITY_KG_M3;
        let dynamic_viscosity = AIR_VISCOSITY_PA_S;
        Self {
            name: "Air".to_string(),
            density,
            dynamic_viscosity,
            kinematic_viscosity: dynamic_viscosity / density,
            specific_heat: 1005.0,
            thermal_conductivity: 0.0257,
        }
    }

    /// Generic oil with user-specified density (kg/m³) and dynamic viscosity (Pa·s).
    pub fn oil(density: f64, viscosity: f64) -> Self {
        Self {
            name: "Oil".to_string(),
            density,
            dynamic_viscosity: viscosity,
            kinematic_viscosity: viscosity / density,
            specific_heat: 1900.0,
            thermal_conductivity: 0.145,
        }
    }
}

// ---------------------------------------------------------------------------
// FlowRegime
// ---------------------------------------------------------------------------

/// Flow regime determined by the Reynolds number.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlowRegime {
    /// Re < 2300
    Laminar,
    /// 2300 ≤ Re < 4000
    Transitional,
    /// Re ≥ 4000
    Turbulent,
}

// ---------------------------------------------------------------------------
// PipeFlowResult
// ---------------------------------------------------------------------------

/// Results of a complete pipe-flow analysis.
#[derive(Debug, Clone)]
pub struct PipeFlowResult {
    /// Reynolds number (dimensionless)
    pub reynolds_number: f64,
    /// Flow regime (Laminar / Transitional / Turbulent)
    pub regime: FlowRegime,
    /// Darcy-Weisbach friction factor (dimensionless)
    pub friction_factor: f64,
    /// Pressure drop over the pipe length in Pa
    pub pressure_drop_pa: f64,
    /// Mean flow velocity in m/s
    pub flow_velocity_m_s: f64,
    /// Head loss in m (= pressure_drop / (ρ g))
    pub head_loss_m: f64,
}

// ---------------------------------------------------------------------------
// FluidError
// ---------------------------------------------------------------------------

/// Errors produced by fluid-dynamics calculations.
#[derive(Debug)]
pub enum FluidError {
    /// An input parameter has an illegal value.
    InvalidInput(String),
    /// An iterative solver did not converge.
    ConvergenceFailure(String),
    /// The requested state is physically impossible.
    PhysicallyImpossible(String),
}

impl std::fmt::Display for FluidError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FluidError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            FluidError::ConvergenceFailure(msg) => write!(f, "Convergence failure: {}", msg),
            FluidError::PhysicallyImpossible(msg) => write!(f, "Physically impossible: {}", msg),
        }
    }
}

impl std::error::Error for FluidError {}

// ---------------------------------------------------------------------------
// FluidDynamics — main calculation engine
// ---------------------------------------------------------------------------

/// Stateless namespace for fluid-dynamics calculations.
pub struct FluidDynamics;

impl FluidDynamics {
    // -----------------------------------------------------------------------
    // Core dimensionless numbers
    // -----------------------------------------------------------------------

    /// Reynolds number: Re = ρ · v · L / μ.
    ///
    /// * `density`   – fluid density in kg/m³
    /// * `velocity`  – flow velocity in m/s
    /// * `length`    – characteristic length (e.g. pipe diameter) in m
    /// * `viscosity` – dynamic viscosity in Pa·s
    pub fn reynolds_number(density: f64, velocity: f64, length: f64, viscosity: f64) -> f64 {
        density * velocity * length / viscosity
    }

    /// Classify a Reynolds number into a `FlowRegime`.
    pub fn flow_regime(re: f64) -> FlowRegime {
        if re < 2300.0 {
            FlowRegime::Laminar
        } else if re < 4000.0 {
            FlowRegime::Transitional
        } else {
            FlowRegime::Turbulent
        }
    }

    // -----------------------------------------------------------------------
    // Friction factor
    // -----------------------------------------------------------------------

    /// Darcy-Weisbach friction factor using the Moody-chart approximation.
    ///
    /// * Laminar  → f = 64 / Re
    /// * Turbulent / Transitional → Swamee-Jain explicit approximation:
    ///   f = 0.25 / (log₁₀(ε/(3.7·D) + 5.74/Re^0.9))²
    ///
    /// # Errors
    /// Returns `FluidError::InvalidInput` if `re` ≤ 0, `diameter` ≤ 0, or
    /// `roughness` < 0.
    pub fn darcy_friction_factor(
        re: f64,
        roughness: f64,
        diameter: f64,
    ) -> Result<f64, FluidError> {
        if re <= 0.0 {
            return Err(FluidError::InvalidInput(
                "Reynolds number must be positive".to_string(),
            ));
        }
        if diameter <= 0.0 {
            return Err(FluidError::InvalidInput(
                "Pipe diameter must be positive".to_string(),
            ));
        }
        if roughness < 0.0 {
            return Err(FluidError::InvalidInput(
                "Surface roughness must be non-negative".to_string(),
            ));
        }

        if re < 2300.0 {
            // Laminar region: exact solution
            return Ok(64.0 / re);
        }

        // Swamee-Jain approximation (turbulent and transitional)
        let relative_roughness = roughness / diameter;
        let arg = relative_roughness / 3.7 + 5.74 / re.powf(0.9);
        if arg <= 0.0 {
            return Err(FluidError::ConvergenceFailure(
                "Swamee-Jain argument is non-positive".to_string(),
            ));
        }
        let log_val = arg.log10();
        Ok(0.25 / (log_val * log_val))
    }

    // -----------------------------------------------------------------------
    // Pressure drop
    // -----------------------------------------------------------------------

    /// Darcy-Weisbach pressure drop: ΔP = f · (L/D) · (ρ · v² / 2).
    ///
    /// * `friction_factor` – Darcy friction factor (dimensionless)
    /// * `length`          – pipe length in m
    /// * `diameter`        – pipe diameter in m
    /// * `density`         – fluid density in kg/m³
    /// * `velocity`        – mean flow velocity in m/s
    pub fn pressure_drop(
        friction_factor: f64,
        length: f64,
        diameter: f64,
        density: f64,
        velocity: f64,
    ) -> f64 {
        friction_factor * (length / diameter) * (density * velocity * velocity / 2.0)
    }

    // -----------------------------------------------------------------------
    // Full pipe-flow analysis
    // -----------------------------------------------------------------------

    /// Perform a complete pipe-flow analysis.
    ///
    /// # Errors
    /// Propagates `FluidError` from `darcy_friction_factor` or returns
    /// `InvalidInput` for non-positive `density`.
    pub fn analyze_pipe_flow(
        fluid: &FluidProperties,
        diameter: f64,
        length: f64,
        velocity: f64,
        roughness: f64,
    ) -> Result<PipeFlowResult, FluidError> {
        if fluid.density <= 0.0 {
            return Err(FluidError::InvalidInput(
                "Fluid density must be positive".to_string(),
            ));
        }
        if diameter <= 0.0 {
            return Err(FluidError::InvalidInput(
                "Diameter must be positive".to_string(),
            ));
        }
        if length <= 0.0 {
            return Err(FluidError::InvalidInput(
                "Pipe length must be positive".to_string(),
            ));
        }

        let re = Self::reynolds_number(fluid.density, velocity, diameter, fluid.dynamic_viscosity);
        let regime = Self::flow_regime(re);
        let friction_factor = Self::darcy_friction_factor(re, roughness, diameter)?;
        let pressure_drop_pa =
            Self::pressure_drop(friction_factor, length, diameter, fluid.density, velocity);
        let head_loss_m = pressure_drop_pa / (fluid.density * GRAVITY_M_S2);

        Ok(PipeFlowResult {
            reynolds_number: re,
            regime,
            friction_factor,
            pressure_drop_pa,
            flow_velocity_m_s: velocity,
            head_loss_m,
        })
    }

    // -----------------------------------------------------------------------
    // Bernoulli
    // -----------------------------------------------------------------------

    /// Solve for p₂ using the Bernoulli equation:
    ///
    /// p₁ + ½ρv₁² + ρgh₁ = p₂ + ½ρv₂² + ρgh₂
    ///
    /// All pressures in Pa, velocities in m/s, heights in m.
    pub fn bernoulli_p2(p1: f64, v1: f64, h1: f64, v2: f64, h2: f64, density: f64) -> f64 {
        let dynamic_1 = 0.5 * density * v1 * v1;
        let dynamic_2 = 0.5 * density * v2 * v2;
        let potential_1 = density * GRAVITY_M_S2 * h1;
        let potential_2 = density * GRAVITY_M_S2 * h2;
        p1 + dynamic_1 + potential_1 - dynamic_2 - potential_2
    }

    // -----------------------------------------------------------------------
    // Continuity
    // -----------------------------------------------------------------------

    /// Continuity equation: v₂ = v₁ · (d₁ / d₂)².
    ///
    /// # Errors
    /// Returns `FluidError::InvalidInput` if either diameter is ≤ 0.
    pub fn continuity_velocity(v1: f64, d1: f64, d2: f64) -> Result<f64, FluidError> {
        if d1 <= 0.0 {
            return Err(FluidError::InvalidInput(
                "Upstream diameter must be positive".to_string(),
            ));
        }
        if d2 <= 0.0 {
            return Err(FluidError::InvalidInput(
                "Downstream diameter must be positive".to_string(),
            ));
        }
        Ok(v1 * (d1 / d2) * (d1 / d2))
    }

    // -----------------------------------------------------------------------
    // Hydraulic diameter
    // -----------------------------------------------------------------------

    /// Hydraulic diameter for non-circular cross-sections: D_h = 4A / P.
    ///
    /// # Errors
    /// Returns `FluidError::InvalidInput` if `area` or `perimeter` is ≤ 0.
    pub fn hydraulic_diameter(area: f64, perimeter: f64) -> Result<f64, FluidError> {
        if area <= 0.0 {
            return Err(FluidError::InvalidInput(
                "Cross-sectional area must be positive".to_string(),
            ));
        }
        if perimeter <= 0.0 {
            return Err(FluidError::InvalidInput(
                "Perimeter must be positive".to_string(),
            ));
        }
        Ok(4.0 * area / perimeter)
    }

    // -----------------------------------------------------------------------
    // Heat transfer
    // -----------------------------------------------------------------------

    /// Nusselt number for turbulent pipe flow (Dittus-Boelter, heating):
    /// Nu = 0.023 · Re^0.8 · Pr^0.4.
    pub fn nusselt_turbulent(re: f64, prandtl: f64) -> f64 {
        0.023 * re.powf(0.8) * prandtl.powf(0.4)
    }

    /// Prandtl number: Pr = μ · Cp / k.
    pub fn prandtl_number(viscosity: f64, specific_heat: f64, thermal_conductivity: f64) -> f64 {
        viscosity * specific_heat / thermal_conductivity
    }

    /// Convective heat transfer coefficient: h = Nu · k / D.
    ///
    /// # Errors
    /// Returns `FluidError::InvalidInput` if `diameter` ≤ 0.
    pub fn convective_htc(
        nusselt: f64,
        conductivity: f64,
        diameter: f64,
    ) -> Result<f64, FluidError> {
        if diameter <= 0.0 {
            return Err(FluidError::InvalidInput(
                "Diameter must be positive".to_string(),
            ));
        }
        Ok(nusselt * conductivity / diameter)
    }

    // -----------------------------------------------------------------------
    // Drag
    // -----------------------------------------------------------------------

    /// Drag force on a sphere: F = Cd · ½ρv² · A, where A = π r².
    pub fn sphere_drag_force(cd: f64, density: f64, velocity: f64, radius: f64) -> f64 {
        let area = std::f64::consts::PI * radius * radius;
        cd * 0.5 * density * velocity * velocity * area
    }

    /// Drag coefficient for a sphere.
    ///
    /// * Stokes (Re ≪ 1):  Cd = 24 / Re
    /// * Otherwise:        three-region approximation
    pub fn sphere_drag_coefficient(re: f64) -> f64 {
        if re <= 0.0 {
            return f64::INFINITY;
        }
        if re < 1.0 {
            // Stokes law
            24.0 / re
        } else if re < 1000.0 {
            // Intermediate region (Schiller-Naumann)
            24.0 / re * (1.0 + 0.15 * re.powf(0.687))
        } else {
            // Newton's law region
            0.44
        }
    }
}

// ---------------------------------------------------------------------------
// ChannelProperties — auxiliary helper (bonus public type)
// ---------------------------------------------------------------------------

/// Named-channel registry for quick lookup by identifier.
#[derive(Debug, Default)]
pub struct FluidRegistry {
    fluids: HashMap<String, FluidProperties>,
}

impl FluidRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            fluids: HashMap::new(),
        }
    }

    /// Register a fluid by name.
    pub fn register(&mut self, fluid: FluidProperties) {
        self.fluids.insert(fluid.name.clone(), fluid);
    }

    /// Look up a fluid by name.
    pub fn get(&self, name: &str) -> Option<&FluidProperties> {
        self.fluids.get(name)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-6;

    // --- FluidProperties ---------------------------------------------------

    #[test]
    fn test_water_properties() {
        let w = FluidProperties::water();
        assert_eq!(w.name, "Water");
        assert!((w.density - 998.2).abs() < EPS);
        assert!((w.dynamic_viscosity - 1.002e-3).abs() < 1e-10);
        let expected_kv = 1.002e-3 / 998.2;
        assert!((w.kinematic_viscosity - expected_kv).abs() < 1e-12);
    }

    #[test]
    fn test_air_properties() {
        let a = FluidProperties::air();
        assert_eq!(a.name, "Air");
        assert!((a.density - 1.204).abs() < EPS);
        assert!((a.dynamic_viscosity - 1.825e-5).abs() < 1e-10);
    }

    #[test]
    fn test_oil_properties() {
        let o = FluidProperties::oil(870.0, 0.05);
        assert_eq!(o.name, "Oil");
        assert!((o.density - 870.0).abs() < EPS);
        assert!((o.kinematic_viscosity - 0.05 / 870.0).abs() < 1e-10);
    }

    // --- Reynolds number ---------------------------------------------------

    #[test]
    fn test_reynolds_number_basic() {
        // Water at 1 m/s through 0.05 m diameter pipe
        let re = FluidDynamics::reynolds_number(998.2, 1.0, 0.05, 1.002e-3);
        // Expected ≈ 49,800
        assert!((re - 49800.0).abs() < 50.0, "re = {}", re);
    }

    #[test]
    fn test_reynolds_number_air() {
        let re = FluidDynamics::reynolds_number(1.204, 5.0, 0.1, 1.825e-5);
        let expected = 1.204 * 5.0 * 0.1 / 1.825e-5;
        assert!((re - expected).abs() / expected < 1e-9);
    }

    #[test]
    fn test_reynolds_number_formula() {
        let re = FluidDynamics::reynolds_number(1000.0, 2.0, 0.02, 0.001);
        assert!((re - 40_000.0).abs() < EPS);
    }

    // --- Flow regime -------------------------------------------------------

    #[test]
    fn test_flow_regime_laminar_boundary() {
        assert_eq!(FluidDynamics::flow_regime(2299.9), FlowRegime::Laminar);
    }

    #[test]
    fn test_flow_regime_laminar_low() {
        assert_eq!(FluidDynamics::flow_regime(100.0), FlowRegime::Laminar);
    }

    #[test]
    fn test_flow_regime_transitional_lower() {
        assert_eq!(FluidDynamics::flow_regime(2300.0), FlowRegime::Transitional);
    }

    #[test]
    fn test_flow_regime_transitional_upper() {
        assert_eq!(FluidDynamics::flow_regime(3999.9), FlowRegime::Transitional);
    }

    #[test]
    fn test_flow_regime_turbulent_boundary() {
        assert_eq!(FluidDynamics::flow_regime(4000.0), FlowRegime::Turbulent);
    }

    #[test]
    fn test_flow_regime_turbulent_high() {
        assert_eq!(FluidDynamics::flow_regime(100_000.0), FlowRegime::Turbulent);
    }

    // --- Friction factor ---------------------------------------------------

    #[test]
    fn test_laminar_friction_factor_64_over_re() {
        let re = 1000.0;
        let f = FluidDynamics::darcy_friction_factor(re, 0.0, 0.05).expect("ok");
        assert!((f - 64.0 / re).abs() < EPS, "f = {}", f);
    }

    #[test]
    fn test_laminar_friction_factor_re_100() {
        let f = FluidDynamics::darcy_friction_factor(100.0, 0.0, 0.05).expect("ok");
        assert!((f - 0.64).abs() < EPS, "f = {}", f);
    }

    #[test]
    fn test_turbulent_friction_swamee_jain() {
        // Smooth pipe, Re = 100_000
        let re = 100_000.0;
        let diameter = 0.1;
        let roughness = 0.0; // smooth
        let f = FluidDynamics::darcy_friction_factor(re, roughness, diameter).expect("ok");
        // Swamee-Jain with ε/D = 0: f = 0.25/(log10(5.74/Re^0.9))^2
        let arg = 5.74 / re.powf(0.9);
        let expected = 0.25 / (arg.log10() * arg.log10());
        assert!((f - expected).abs() / expected < 1e-9, "f = {}", f);
    }

    #[test]
    fn test_turbulent_rough_pipe() {
        let f = FluidDynamics::darcy_friction_factor(50_000.0, 4.6e-5, 0.05).expect("ok");
        // Sanity: turbulent f for commercial steel is roughly 0.02–0.04
        assert!(f > 0.01 && f < 0.1, "f out of reasonable range: {}", f);
    }

    #[test]
    fn test_friction_factor_invalid_re() {
        assert!(FluidDynamics::darcy_friction_factor(0.0, 0.0, 0.05).is_err());
        assert!(FluidDynamics::darcy_friction_factor(-1.0, 0.0, 0.05).is_err());
    }

    #[test]
    fn test_friction_factor_invalid_diameter() {
        assert!(FluidDynamics::darcy_friction_factor(1000.0, 0.0, 0.0).is_err());
        assert!(FluidDynamics::darcy_friction_factor(1000.0, 0.0, -0.1).is_err());
    }

    #[test]
    fn test_friction_factor_negative_roughness() {
        assert!(FluidDynamics::darcy_friction_factor(10_000.0, -0.001, 0.05).is_err());
    }

    // --- Pressure drop -----------------------------------------------------

    #[test]
    fn test_pressure_drop_formula() {
        // ΔP = f * (L/D) * (ρ v²/2)
        let dp = FluidDynamics::pressure_drop(0.02, 10.0, 0.05, 1000.0, 1.0);
        let expected = 0.02 * (10.0 / 0.05) * (1000.0 * 1.0 / 2.0);
        assert!((dp - expected).abs() < EPS, "dp = {}", dp);
    }

    #[test]
    fn test_pressure_drop_zero_velocity() {
        let dp = FluidDynamics::pressure_drop(0.03, 5.0, 0.1, 1000.0, 0.0);
        assert!((dp).abs() < EPS);
    }

    // --- Full pipe-flow analysis -------------------------------------------

    #[test]
    fn test_analyze_pipe_flow_water_laminar() {
        let water = FluidProperties::water();
        // Low velocity → laminar
        let result = FluidDynamics::analyze_pipe_flow(&water, 0.05, 10.0, 0.01, 0.0)
            .expect("should succeed");
        assert_eq!(result.regime, FlowRegime::Laminar);
        assert!(result.reynolds_number < 2300.0);
        let expected_f = 64.0 / result.reynolds_number;
        assert!((result.friction_factor - expected_f).abs() < EPS);
    }

    #[test]
    fn test_analyze_pipe_flow_water_turbulent() {
        let water = FluidProperties::water();
        let result = FluidDynamics::analyze_pipe_flow(&water, 0.05, 10.0, 2.0, 4.6e-5)
            .expect("should succeed");
        assert_eq!(result.regime, FlowRegime::Turbulent);
        assert!(result.pressure_drop_pa > 0.0);
        assert!(result.head_loss_m > 0.0);
        // head_loss = pressure_drop / (density * g)
        let expected_hl = result.pressure_drop_pa / (water.density * GRAVITY_M_S2);
        assert!((result.head_loss_m - expected_hl).abs() < 1e-6);
    }

    #[test]
    fn test_analyze_pipe_flow_air() {
        let air = FluidProperties::air();
        let result =
            FluidDynamics::analyze_pipe_flow(&air, 0.1, 5.0, 10.0, 0.0).expect("should succeed");
        assert!(result.pressure_drop_pa > 0.0);
    }

    #[test]
    fn test_analyze_pipe_flow_invalid_density() {
        let mut bad_fluid = FluidProperties::water();
        bad_fluid.density = -1.0;
        assert!(FluidDynamics::analyze_pipe_flow(&bad_fluid, 0.05, 10.0, 1.0, 0.0).is_err());
    }

    #[test]
    fn test_analyze_pipe_flow_invalid_diameter() {
        let water = FluidProperties::water();
        assert!(FluidDynamics::analyze_pipe_flow(&water, 0.0, 10.0, 1.0, 0.0).is_err());
    }

    #[test]
    fn test_analyze_pipe_flow_invalid_length() {
        let water = FluidProperties::water();
        assert!(FluidDynamics::analyze_pipe_flow(&water, 0.05, 0.0, 1.0, 0.0).is_err());
    }

    // --- Bernoulli ----------------------------------------------------------

    #[test]
    fn test_bernoulli_p2_same_velocity_height() {
        // v1 = v2, h1 = h2 → p2 = p1
        let p2 = FluidDynamics::bernoulli_p2(101_325.0, 2.0, 0.0, 2.0, 0.0, 1000.0);
        assert!((p2 - 101_325.0).abs() < EPS);
    }

    #[test]
    fn test_bernoulli_p2_speed_increase() {
        // Faster downstream → lower pressure
        let p2 = FluidDynamics::bernoulli_p2(200_000.0, 1.0, 0.0, 3.0, 0.0, 1000.0);
        assert!(p2 < 200_000.0);
    }

    #[test]
    fn test_bernoulli_p2_height_increase() {
        // Higher elevation → lower pressure (same velocity)
        let p2 = FluidDynamics::bernoulli_p2(200_000.0, 1.0, 0.0, 1.0, 5.0, 1000.0);
        let expected = 200_000.0 - 1000.0 * GRAVITY_M_S2 * 5.0;
        assert!((p2 - expected).abs() < EPS);
    }

    #[test]
    fn test_bernoulli_p2_formula() {
        let rho = 998.2;
        let p2 = FluidDynamics::bernoulli_p2(100_000.0, 2.0, 1.0, 4.0, 2.0, rho);
        let expected = 100_000.0
            + 0.5 * rho * (2.0_f64.powi(2) - 4.0_f64.powi(2))
            + rho * GRAVITY_M_S2 * (1.0 - 2.0);
        assert!((p2 - expected).abs() < 1e-3);
    }

    // --- Continuity --------------------------------------------------------

    #[test]
    fn test_continuity_velocity_same_diameter() {
        let v2 = FluidDynamics::continuity_velocity(5.0, 0.1, 0.1).expect("ok");
        assert!((v2 - 5.0).abs() < EPS);
    }

    #[test]
    fn test_continuity_velocity_halved_diameter() {
        // d2 = d1/2 → A ratio = 4 → v2 = 4 * v1
        let v2 = FluidDynamics::continuity_velocity(2.0, 0.1, 0.05).expect("ok");
        assert!((v2 - 8.0).abs() < EPS, "v2 = {}", v2);
    }

    #[test]
    fn test_continuity_velocity_area_ratio() {
        let v1 = 3.0;
        let d1 = 0.2;
        let d2 = 0.1;
        let v2 = FluidDynamics::continuity_velocity(v1, d1, d2).expect("ok");
        assert!((v2 - v1 * (d1 / d2).powi(2)).abs() < EPS);
    }

    #[test]
    fn test_continuity_velocity_zero_d1() {
        assert!(FluidDynamics::continuity_velocity(1.0, 0.0, 0.05).is_err());
    }

    #[test]
    fn test_continuity_velocity_zero_d2() {
        assert!(FluidDynamics::continuity_velocity(1.0, 0.05, 0.0).is_err());
    }

    // --- Hydraulic diameter ------------------------------------------------

    #[test]
    fn test_hydraulic_diameter_circular() {
        // Circle: A = π r², P = 2π r → D_h = 4·π r²/(2π r) = 2r = diameter
        let r = 0.05_f64;
        let area = std::f64::consts::PI * r * r;
        let perimeter = 2.0 * std::f64::consts::PI * r;
        let dh = FluidDynamics::hydraulic_diameter(area, perimeter).expect("ok");
        assert!((dh - 2.0 * r).abs() < 1e-10, "dh = {}", dh);
    }

    #[test]
    fn test_hydraulic_diameter_square() {
        // Square with side a: A = a², P = 4a → D_h = 4a²/4a = a
        let a = 0.1_f64;
        let dh = FluidDynamics::hydraulic_diameter(a * a, 4.0 * a).expect("ok");
        assert!((dh - a).abs() < EPS, "dh = {}", dh);
    }

    #[test]
    fn test_hydraulic_diameter_invalid_area() {
        assert!(FluidDynamics::hydraulic_diameter(0.0, 0.5).is_err());
    }

    #[test]
    fn test_hydraulic_diameter_invalid_perimeter() {
        assert!(FluidDynamics::hydraulic_diameter(0.01, 0.0).is_err());
    }

    // --- Nusselt number ----------------------------------------------------

    #[test]
    fn test_nusselt_turbulent_formula() {
        let re = 10_000.0;
        let pr = 7.0;
        let nu = FluidDynamics::nusselt_turbulent(re, pr);
        let expected = 0.023 * re.powf(0.8) * pr.powf(0.4);
        assert!((nu - expected).abs() / expected < 1e-9);
    }

    #[test]
    fn test_nusselt_turbulent_water() {
        // Typical water pipe flow
        let nu = FluidDynamics::nusselt_turbulent(50_000.0, 6.99);
        assert!(nu > 100.0, "nu = {}", nu);
    }

    // --- Prandtl number ----------------------------------------------------

    #[test]
    fn test_prandtl_number_water() {
        let pr = FluidDynamics::prandtl_number(1.002e-3, 4182.0, 0.598);
        // Expected ≈ 7.0
        assert!((pr - 7.0).abs() < 0.1, "Pr = {}", pr);
    }

    #[test]
    fn test_prandtl_number_air() {
        let pr = FluidDynamics::prandtl_number(1.825e-5, 1005.0, 0.0257);
        // Expected ≈ 0.71
        assert!((pr - 0.71).abs() < 0.01, "Pr = {}", pr);
    }

    #[test]
    fn test_prandtl_number_formula() {
        let mu = 0.002;
        let cp = 2000.0;
        let k = 0.5;
        let pr = FluidDynamics::prandtl_number(mu, cp, k);
        assert!((pr - mu * cp / k).abs() < EPS);
    }

    // --- Convective HTC ----------------------------------------------------

    #[test]
    fn test_convective_htc_basic() {
        let h = FluidDynamics::convective_htc(200.0, 0.598, 0.05).expect("ok");
        let expected = 200.0 * 0.598 / 0.05;
        assert!((h - expected).abs() < EPS);
    }

    #[test]
    fn test_convective_htc_invalid_diameter() {
        assert!(FluidDynamics::convective_htc(100.0, 0.5, 0.0).is_err());
        assert!(FluidDynamics::convective_htc(100.0, 0.5, -0.01).is_err());
    }

    // --- Drag ---------------------------------------------------------------

    #[test]
    fn test_sphere_drag_force_formula() {
        let cd = 0.44;
        let rho = 1.204;
        let v = 10.0;
        let r = 0.05;
        let f = FluidDynamics::sphere_drag_force(cd, rho, v, r);
        let area = std::f64::consts::PI * r * r;
        let expected = cd * 0.5 * rho * v * v * area;
        assert!((f - expected).abs() / expected < 1e-9);
    }

    #[test]
    fn test_sphere_drag_force_zero_velocity() {
        let f = FluidDynamics::sphere_drag_force(0.44, 1.204, 0.0, 0.05);
        assert!(f.abs() < EPS);
    }

    #[test]
    fn test_sphere_drag_coefficient_stokes() {
        let re = 0.5;
        let cd = FluidDynamics::sphere_drag_coefficient(re);
        let expected = 24.0 / re;
        assert!((cd - expected).abs() < EPS, "cd = {}", cd);
    }

    #[test]
    fn test_sphere_drag_coefficient_newton_region() {
        let cd = FluidDynamics::sphere_drag_coefficient(100_000.0);
        assert!((cd - 0.44).abs() < EPS);
    }

    #[test]
    fn test_sphere_drag_coefficient_intermediate() {
        let re = 100.0;
        let cd = FluidDynamics::sphere_drag_coefficient(re);
        // Should be larger than Newton value but smaller than Stokes for this Re
        assert!(cd > 0.44 && cd < 24.0 / re + 1.0);
    }

    #[test]
    fn test_sphere_drag_coefficient_zero_re() {
        let cd = FluidDynamics::sphere_drag_coefficient(0.0);
        assert!(cd.is_infinite());
    }

    // --- Registry ----------------------------------------------------------

    #[test]
    fn test_fluid_registry() {
        let mut reg = FluidRegistry::new();
        reg.register(FluidProperties::water());
        reg.register(FluidProperties::air());
        assert!(reg.get("Water").is_some());
        assert!(reg.get("Air").is_some());
        assert!(reg.get("Ethanol").is_none());
    }

    // --- Error display -----------------------------------------------------

    #[test]
    fn test_fluid_error_display() {
        let e = FluidError::InvalidInput("bad value".to_string());
        assert!(e.to_string().contains("bad value"));
        let e2 = FluidError::ConvergenceFailure("too slow".to_string());
        assert!(e2.to_string().contains("too slow"));
        let e3 = FluidError::PhysicallyImpossible("negative density".to_string());
        assert!(e3.to_string().contains("negative density"));
    }

    // --- Additional edge-case tests ----------------------------------------

    #[test]
    fn test_pressure_drop_scaling_with_velocity() {
        // Doubling velocity quadruples pressure drop (ΔP ∝ v²)
        let dp1 = FluidDynamics::pressure_drop(0.025, 10.0, 0.05, 1000.0, 2.0);
        let dp2 = FluidDynamics::pressure_drop(0.025, 10.0, 0.05, 1000.0, 4.0);
        assert!((dp2 / dp1 - 4.0).abs() < 1e-9, "ratio = {}", dp2 / dp1);
    }

    #[test]
    fn test_bernoulli_venturi_tube() {
        // Venturi tube: constriction d2 = 0.5 d1 → v2 = 4 v1
        let rho = 1000.0;
        let v1 = 1.0;
        let v2 = FluidDynamics::continuity_velocity(v1, 0.1, 0.05).expect("ok");
        assert!((v2 - 4.0 * v1).abs() < EPS);
        let p2 = FluidDynamics::bernoulli_p2(200_000.0, v1, 0.0, v2, 0.0, rho);
        // Pressure must drop in the throat
        assert!(p2 < 200_000.0);
    }

    #[test]
    fn test_head_loss_nondimensional_consistency() {
        let water = FluidProperties::water();
        let result =
            FluidDynamics::analyze_pipe_flow(&water, 0.025, 100.0, 1.5, 4.6e-5).expect("ok");
        // Verify head_loss from formula
        let hl_check = result.pressure_drop_pa / (water.density * GRAVITY_M_S2);
        assert!((result.head_loss_m - hl_check).abs() < 1e-6);
    }
}
