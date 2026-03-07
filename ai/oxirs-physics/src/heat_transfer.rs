//! # Heat Transfer Simulation
//!
//! Provides fundamental heat transfer calculations covering conduction (Fourier's law),
//! convection (Newton's law of cooling), and radiation (Stefan-Boltzmann law).
//!
//! # Examples
//!
//! ```rust
//! use oxirs_physics::heat_transfer::{Material, ThermalLayer, HeatTransferCalc};
//!
//! let mat = Material::COPPER;
//! let layer = ThermalLayer { material: mat, thickness: 0.01, area: 1.0 };
//! let flux = HeatTransferCalc::conduction_heat_flux(&layer.material, 100.0, 20.0, 0.01);
//! assert!((flux - 3_080_000.0).abs() < 1.0);
//! ```

/// Stefan-Boltzmann constant (W/m²·K⁴)
pub const STEFAN_BOLTZMANN: f64 = 5.67e-8;

/// Heat transfer mode
#[derive(Debug, Clone, PartialEq)]
pub enum HeatTransferMode {
    /// Fourier conduction through a solid medium
    Conduction,
    /// Newton convection between a surface and fluid
    Convection,
    /// Stefan-Boltzmann thermal radiation
    Radiation,
    /// All three modes simultaneously
    Combined,
}

/// Thermal material properties
#[derive(Debug, Clone, PartialEq)]
pub struct Material {
    /// Human-readable material name
    pub name: String,
    /// Thermal conductivity k (W/m·K)
    pub thermal_conductivity: f64,
    /// Mass density ρ (kg/m³)
    pub density: f64,
    /// Specific heat capacity c_p (J/kg·K)
    pub specific_heat: f64,
}

impl Material {
    /// Copper — excellent electrical and thermal conductor
    pub const COPPER: Material = Material {
        name: String::new(), // replaced in impl below via a const fn approach
        thermal_conductivity: 385.0,
        density: 8_960.0,
        specific_heat: 385.0,
    };

    /// Structural (carbon) steel
    pub const STEEL: Material = Material {
        name: String::new(),
        thermal_conductivity: 50.0,
        density: 7_850.0,
        specific_heat: 490.0,
    };

    /// Normal-weight concrete
    pub const CONCRETE: Material = Material {
        name: String::new(),
        thermal_conductivity: 1.0,
        density: 2_300.0,
        specific_heat: 880.0,
    };

    /// Dry softwood (pine)
    pub const WOOD: Material = Material {
        name: String::new(),
        thermal_conductivity: 0.12,
        density: 550.0,
        specific_heat: 1_700.0,
    };

    /// Dry air at ~25 °C, 1 atm
    pub const AIR: Material = Material {
        name: String::new(),
        thermal_conductivity: 0.026,
        density: 1.184,
        specific_heat: 1_005.0,
    };

    /// Return a named copper instance.
    pub fn copper() -> Self {
        Self {
            name: "Copper".to_string(),
            ..Self::COPPER
        }
    }

    /// Return a named steel instance.
    pub fn steel() -> Self {
        Self {
            name: "Steel".to_string(),
            ..Self::STEEL
        }
    }

    /// Return a named concrete instance.
    pub fn concrete() -> Self {
        Self {
            name: "Concrete".to_string(),
            ..Self::CONCRETE
        }
    }

    /// Return a named wood instance.
    pub fn wood() -> Self {
        Self {
            name: "Wood".to_string(),
            ..Self::WOOD
        }
    }

    /// Return a named air instance.
    pub fn air() -> Self {
        Self {
            name: "Air".to_string(),
            ..Self::AIR
        }
    }

    /// Thermal diffusivity α = k / (ρ · c_p)  (m²/s)
    pub fn thermal_diffusivity(&self) -> f64 {
        self.thermal_conductivity / (self.density * self.specific_heat)
    }
}

/// A single homogeneous thermal layer with geometry
#[derive(Debug, Clone)]
pub struct ThermalLayer {
    /// Material properties
    pub material: Material,
    /// Layer thickness (m)
    pub thickness: f64,
    /// Cross-sectional area (m²)
    pub area: f64,
}

impl ThermalLayer {
    /// Thermal resistance of this layer: R = L / (k · A)  (K/W)
    pub fn thermal_resistance(&self) -> f64 {
        self.thickness / (self.material.thermal_conductivity * self.area)
    }
}

/// Stateless collection of heat-transfer calculation functions
pub struct HeatTransferCalc;

impl HeatTransferCalc {
    // -----------------------------------------------------------------------
    // Conduction
    // -----------------------------------------------------------------------

    /// Fourier heat flux: q = k · ΔT / L  (W/m²)
    ///
    /// * `material`  – layer material
    /// * `temp_hot`  – hot-side temperature (°C or K; only difference matters)
    /// * `temp_cold` – cold-side temperature
    /// * `thickness` – layer thickness (m)
    pub fn conduction_heat_flux(
        material: &Material,
        temp_hot: f64,
        temp_cold: f64,
        thickness: f64,
    ) -> f64 {
        if thickness <= 0.0 {
            return 0.0;
        }
        material.thermal_conductivity * (temp_hot - temp_cold) / thickness
    }

    /// Conduction power Q = q · A  (W) through a single layer
    pub fn conduction_power(layer: &ThermalLayer, temp_hot: f64, temp_cold: f64) -> f64 {
        let flux =
            Self::conduction_heat_flux(&layer.material, temp_hot, temp_cold, layer.thickness);
        flux * layer.area
    }

    // -----------------------------------------------------------------------
    // Composite wall — resistance network
    // -----------------------------------------------------------------------

    /// Total thermal resistance for layers in *series*: R = Σ (L_i / (k_i · A_i))  (K/W)
    pub fn series_resistance(layers: &[ThermalLayer]) -> f64 {
        layers.iter().map(|l| l.thermal_resistance()).sum()
    }

    /// Total thermal resistance for layers in *parallel*: 1/R = Σ (1/R_i)  (K/W)
    pub fn parallel_resistance(layers: &[ThermalLayer]) -> f64 {
        let inv_sum: f64 = layers.iter().map(|l| 1.0 / l.thermal_resistance()).sum();
        if inv_sum == 0.0 {
            f64::INFINITY
        } else {
            1.0 / inv_sum
        }
    }

    /// Heat flow through layers in series: Q = ΔT / R_total  (W)
    pub fn heat_flow_series(layers: &[ThermalLayer], temp_hot: f64, temp_cold: f64) -> f64 {
        let r = Self::series_resistance(layers);
        if r == 0.0 {
            return 0.0;
        }
        (temp_hot - temp_cold) / r
    }

    // -----------------------------------------------------------------------
    // Convection
    // -----------------------------------------------------------------------

    /// Newton's law of convection: q = h · (T_s - T_f)  (W/m²)
    ///
    /// * `h`            – convective heat transfer coefficient (W/m²·K)
    /// * `temp_surface` – surface temperature
    /// * `temp_fluid`   – bulk fluid temperature
    pub fn convection_heat_flux(h: f64, temp_surface: f64, temp_fluid: f64) -> f64 {
        h * (temp_surface - temp_fluid)
    }

    // -----------------------------------------------------------------------
    // Radiation
    // -----------------------------------------------------------------------

    /// Stefan-Boltzmann radiation flux: q = ε·σ·(T_s⁴ - T_a⁴)  (W/m²)
    ///
    /// Temperatures **must** be in Kelvin.
    ///
    /// * `emissivity`    – surface emissivity (0–1)
    /// * `temp_surface`  – surface temperature (K)
    /// * `temp_ambient`  – ambient temperature (K)
    pub fn radiation_heat_flux(emissivity: f64, temp_surface: f64, temp_ambient: f64) -> f64 {
        emissivity * STEFAN_BOLTZMANN * (temp_surface.powi(4) - temp_ambient.powi(4))
    }

    // -----------------------------------------------------------------------
    // Combined surface loss
    // -----------------------------------------------------------------------

    /// Total heat loss from a surface by convection + radiation (W)
    ///
    /// * `layer`        – surface layer geometry (area used for power)
    /// * `h_conv`       – convective coefficient (W/m²·K)
    /// * `emissivity`   – surface emissivity
    /// * `temp_surface` – surface temperature (K)
    /// * `temp_ambient` – ambient temperature (K)
    pub fn combined_heat_loss(
        layer: &ThermalLayer,
        h_conv: f64,
        emissivity: f64,
        temp_surface: f64,
        temp_ambient: f64,
    ) -> f64 {
        let q_conv = Self::convection_heat_flux(h_conv, temp_surface, temp_ambient);
        let q_rad = Self::radiation_heat_flux(emissivity, temp_surface, temp_ambient);
        (q_conv + q_rad) * layer.area
    }

    // -----------------------------------------------------------------------
    // Transient estimate
    // -----------------------------------------------------------------------

    /// Estimate the time for the mid-plane of a 1-D slab to reach `target_temp`.
    ///
    /// Uses the first-term Fourier series approximation for a symmetric slab with
    /// Dirichlet boundary condition (isothermal walls):
    ///
    ///   θ(0, t) ≈ (4/π) · exp(−π²·Fo)  →  t = L² / (π²·α) · ln(4·θ_0 / (π·θ))
    ///
    /// where θ = (T − T_boundary) / (T_init − T_boundary), L is the half-thickness.
    ///
    /// Returns `f64::NAN` if the target is already at or beyond the boundary temperature,
    /// or if the geometry/physics is degenerate.
    ///
    /// * `material`       – slab material
    /// * `thickness`      – full slab thickness (m); half-thickness = thickness/2
    /// * `temp_initial`   – uniform initial temperature
    /// * `temp_boundary`  – constant wall temperature (boundary condition)
    /// * `target_temp`    – desired centre temperature
    pub fn time_to_equilibrium(
        material: &Material,
        thickness: f64,
        temp_initial: f64,
        temp_boundary: f64,
        target_temp: f64,
    ) -> f64 {
        let alpha = material.thermal_diffusivity();
        if alpha <= 0.0 || thickness <= 0.0 {
            return f64::NAN;
        }

        let half = thickness / 2.0;
        let delta_init = temp_initial - temp_boundary;
        let delta_target = target_temp - temp_boundary;

        // Degenerate / already reached
        if delta_init.abs() < 1e-12 || delta_target.abs() < 1e-12 {
            return 0.0;
        }

        // Ensure the target is between initial and boundary
        let ratio = delta_target / delta_init;
        if ratio <= 0.0 || ratio >= 1.0 {
            return f64::NAN;
        }

        // θ = delta_target / delta_init
        // First-term approximation: θ ≈ (4/π) exp(−π²·α·t / L²)
        // ⇒ t = (L² / (π²·α)) · ln( (4/π) / θ )
        let t = (half * half / (std::f64::consts::PI * std::f64::consts::PI * alpha))
            * ((4.0 / std::f64::consts::PI) / ratio).ln();

        if t < 0.0 {
            0.0
        } else {
            t
        }
    }

    // -----------------------------------------------------------------------
    // Interface temperatures
    // -----------------------------------------------------------------------

    /// Compute temperatures at each interface in a series composite wall.
    ///
    /// Returns a `Vec` with `N-1` values for `N` layers (interior interfaces).
    /// If there is only one layer there are no interior interfaces → empty Vec.
    ///
    /// Uses the fact that heat flux Q is uniform in series: T_i = T_{i-1} − Q·R_i
    pub fn interface_temperature(
        layers: &[ThermalLayer],
        temp_hot: f64,
        temp_cold: f64,
    ) -> Vec<f64> {
        if layers.len() < 2 {
            return Vec::new();
        }

        let q = Self::heat_flow_series(layers, temp_hot, temp_cold);
        let mut interfaces = Vec::with_capacity(layers.len() - 1);
        let mut t_current = temp_hot;

        for layer in layers.iter().take(layers.len() - 1) {
            t_current -= q * layer.thermal_resistance();
            interfaces.push(t_current);
        }

        interfaces
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-6;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // --- Material properties ---

    #[test]
    fn test_copper_diffusivity() {
        let m = Material::copper();
        // α = 385 / (8960 * 385) ≈ 1.116e-4 m²/s
        let expected = 385.0 / (8_960.0 * 385.0);
        assert!(approx_eq(m.thermal_diffusivity(), expected, EPS));
    }

    #[test]
    fn test_steel_diffusivity() {
        let m = Material::steel();
        let expected = 50.0 / (7_850.0 * 490.0);
        assert!(approx_eq(m.thermal_diffusivity(), expected, EPS));
    }

    #[test]
    fn test_concrete_properties() {
        let m = Material::concrete();
        assert_eq!(m.thermal_conductivity, 1.0);
        assert_eq!(m.density, 2_300.0);
        assert_eq!(m.specific_heat, 880.0);
    }

    #[test]
    fn test_wood_properties() {
        let m = Material::wood();
        assert_eq!(m.thermal_conductivity, 0.12);
        assert_eq!(m.density, 550.0);
    }

    #[test]
    fn test_air_properties() {
        let m = Material::air();
        assert_eq!(m.thermal_conductivity, 0.026);
    }

    #[test]
    fn test_material_name() {
        assert_eq!(Material::copper().name, "Copper");
        assert_eq!(Material::steel().name, "Steel");
        assert_eq!(Material::concrete().name, "Concrete");
        assert_eq!(Material::wood().name, "Wood");
        assert_eq!(Material::air().name, "Air");
    }

    // --- Conduction flux ---

    #[test]
    fn test_conduction_heat_flux_copper() {
        // q = 385 * (100 - 20) / 0.01 = 3_080_000 W/m²
        let flux = HeatTransferCalc::conduction_heat_flux(&Material::copper(), 100.0, 20.0, 0.01);
        assert!(approx_eq(flux, 3_080_000.0, 1.0));
    }

    #[test]
    fn test_conduction_heat_flux_zero_thickness() {
        let flux = HeatTransferCalc::conduction_heat_flux(&Material::copper(), 100.0, 20.0, 0.0);
        assert_eq!(flux, 0.0);
    }

    #[test]
    fn test_conduction_heat_flux_negative_thickness() {
        let flux = HeatTransferCalc::conduction_heat_flux(&Material::copper(), 100.0, 20.0, -0.01);
        assert_eq!(flux, 0.0);
    }

    #[test]
    fn test_conduction_heat_flux_air() {
        // q = 0.026 * (50 - 20) / 0.1 = 7.8 W/m²
        let flux = HeatTransferCalc::conduction_heat_flux(&Material::air(), 50.0, 20.0, 0.1);
        assert!(approx_eq(flux, 7.8, EPS));
    }

    #[test]
    fn test_conduction_power() {
        let layer = ThermalLayer {
            material: Material::steel(),
            thickness: 0.05,
            area: 2.0,
        };
        // flux = 50 * (200 - 25) / 0.05 = 175_000 W/m²; power = 175_000 * 2 = 350_000 W
        let power = HeatTransferCalc::conduction_power(&layer, 200.0, 25.0);
        assert!(approx_eq(power, 350_000.0, 1.0));
    }

    // --- Thermal resistance ---

    #[test]
    fn test_thermal_resistance_single_layer() {
        let layer = ThermalLayer {
            material: Material::concrete(),
            thickness: 0.2,
            area: 10.0,
        };
        // R = 0.2 / (1.0 * 10) = 0.02 K/W
        assert!(approx_eq(layer.thermal_resistance(), 0.02, EPS));
    }

    #[test]
    fn test_series_resistance_two_layers() {
        let l1 = ThermalLayer {
            material: Material::concrete(),
            thickness: 0.2,
            area: 10.0,
        }; // R = 0.02
        let l2 = ThermalLayer {
            material: Material::wood(),
            thickness: 0.05,
            area: 10.0,
        }; // R = 0.05/1.2 ≈ 0.04167
        let r = HeatTransferCalc::series_resistance(&[l1, l2]);
        assert!(approx_eq(r, 0.02 + 0.05 / (0.12 * 10.0), EPS));
    }

    #[test]
    fn test_parallel_resistance_two_layers() {
        let l1 = ThermalLayer {
            material: Material::concrete(),
            thickness: 0.1,
            area: 5.0,
        }; // R1 = 0.1 / (1*5) = 0.02
        let l2 = ThermalLayer {
            material: Material::concrete(),
            thickness: 0.1,
            area: 5.0,
        }; // R2 = 0.02
        let r = HeatTransferCalc::parallel_resistance(&[l1, l2]);
        // 1/R = 1/0.02 + 1/0.02 = 100; R = 0.01
        assert!(approx_eq(r, 0.01, EPS));
    }

    #[test]
    fn test_heat_flow_series() {
        let layer = ThermalLayer {
            material: Material::concrete(),
            thickness: 0.2,
            area: 10.0,
        }; // R = 0.02 K/W
           // Q = (100 - 20) / 0.02 = 4000 W
        let q = HeatTransferCalc::heat_flow_series(&[layer], 100.0, 20.0);
        assert!(approx_eq(q, 4_000.0, EPS));
    }

    #[test]
    fn test_heat_flow_series_empty() {
        let q = HeatTransferCalc::heat_flow_series(&[], 100.0, 20.0);
        assert_eq!(q, 0.0);
    }

    // --- Convection ---

    #[test]
    fn test_convection_heat_flux() {
        // q = 25 * (80 - 20) = 1500 W/m²
        let flux = HeatTransferCalc::convection_heat_flux(25.0, 80.0, 20.0);
        assert!(approx_eq(flux, 1_500.0, EPS));
    }

    #[test]
    fn test_convection_heat_flux_negative_diff() {
        let flux = HeatTransferCalc::convection_heat_flux(10.0, 15.0, 25.0);
        assert!(approx_eq(flux, -100.0, EPS));
    }

    // --- Radiation ---

    #[test]
    fn test_radiation_heat_flux_blackbody() {
        // Perfect blackbody (ε=1) at 500 K radiating to 300 K ambient
        let flux = HeatTransferCalc::radiation_heat_flux(1.0, 500.0, 300.0);
        let expected = STEFAN_BOLTZMANN * (500.0_f64.powi(4) - 300.0_f64.powi(4));
        assert!(approx_eq(flux, expected, 1e-3));
    }

    #[test]
    fn test_radiation_heat_flux_partial_emissivity() {
        let flux = HeatTransferCalc::radiation_heat_flux(0.5, 400.0, 300.0);
        let expected = 0.5 * STEFAN_BOLTZMANN * (400.0_f64.powi(4) - 300.0_f64.powi(4));
        assert!(approx_eq(flux, expected, 1e-3));
    }

    #[test]
    fn test_radiation_heat_flux_zero_emissivity() {
        let flux = HeatTransferCalc::radiation_heat_flux(0.0, 500.0, 300.0);
        assert_eq!(flux, 0.0);
    }

    #[test]
    fn test_radiation_heat_flux_equal_temps() {
        let flux = HeatTransferCalc::radiation_heat_flux(0.9, 300.0, 300.0);
        assert!(approx_eq(flux, 0.0, EPS));
    }

    // --- Combined ---

    #[test]
    fn test_combined_heat_loss() {
        let layer = ThermalLayer {
            material: Material::steel(),
            thickness: 0.01,
            area: 1.0,
        };
        // h=10, ε=0.8, T_s=400 K, T_a=300 K
        let q_conv = HeatTransferCalc::convection_heat_flux(10.0, 400.0, 300.0); // = 1000 W/m²
        let q_rad = HeatTransferCalc::radiation_heat_flux(0.8, 400.0, 300.0);
        let expected = (q_conv + q_rad) * 1.0;
        let actual = HeatTransferCalc::combined_heat_loss(&layer, 10.0, 0.8, 400.0, 300.0);
        assert!(approx_eq(actual, expected, EPS));
    }

    #[test]
    fn test_combined_heat_loss_area_scaling() {
        let layer1 = ThermalLayer {
            material: Material::steel(),
            thickness: 0.01,
            area: 1.0,
        };
        let layer2 = ThermalLayer {
            material: Material::steel(),
            thickness: 0.01,
            area: 2.0,
        };
        let q1 = HeatTransferCalc::combined_heat_loss(&layer1, 5.0, 0.9, 350.0, 300.0);
        let q2 = HeatTransferCalc::combined_heat_loss(&layer2, 5.0, 0.9, 350.0, 300.0);
        assert!(approx_eq(q2, 2.0 * q1, 1e-6));
    }

    // --- Time to equilibrium ---

    #[test]
    fn test_time_to_equilibrium_basic() {
        let m = Material::copper();
        // Half-slab: copper, L=0.01 m, T_init=200, T_wall=20, target=21 (near boundary)
        let t = HeatTransferCalc::time_to_equilibrium(&m, 0.02, 200.0, 20.0, 21.0);
        assert!(t > 0.0, "time must be positive, got {t}");
        assert!(t.is_finite(), "time must be finite");
    }

    #[test]
    fn test_time_to_equilibrium_already_at_target() {
        let m = Material::copper();
        let t = HeatTransferCalc::time_to_equilibrium(&m, 0.02, 100.0, 20.0, 20.0);
        assert_eq!(t, 0.0);
    }

    #[test]
    fn test_time_to_equilibrium_degenerate_thickness() {
        let m = Material::copper();
        let t = HeatTransferCalc::time_to_equilibrium(&m, 0.0, 200.0, 20.0, 100.0);
        assert!(t.is_nan());
    }

    #[test]
    fn test_time_to_equilibrium_ordering() {
        // A thicker slab takes longer
        let m = Material::concrete();
        let t1 = HeatTransferCalc::time_to_equilibrium(&m, 0.10, 100.0, 0.0, 50.0);
        let t2 = HeatTransferCalc::time_to_equilibrium(&m, 0.20, 100.0, 0.0, 50.0);
        assert!(t2 > t1, "thicker slab should take longer: t1={t1} t2={t2}");
    }

    #[test]
    fn test_time_to_equilibrium_target_beyond_boundary() {
        let m = Material::copper();
        // target below boundary → NaN
        let t = HeatTransferCalc::time_to_equilibrium(&m, 0.02, 200.0, 20.0, 10.0);
        assert!(t.is_nan());
    }

    // --- Interface temperatures ---

    #[test]
    fn test_interface_temperature_single_layer() {
        let layer = ThermalLayer {
            material: Material::concrete(),
            thickness: 0.1,
            area: 1.0,
        };
        let interfaces = HeatTransferCalc::interface_temperature(&[layer], 100.0, 20.0);
        assert!(interfaces.is_empty());
    }

    #[test]
    fn test_interface_temperature_two_layers() {
        // Two identical concrete layers → interface must be at mid-temperature
        let l1 = ThermalLayer {
            material: Material::concrete(),
            thickness: 0.1,
            area: 1.0,
        };
        let l2 = ThermalLayer {
            material: Material::concrete(),
            thickness: 0.1,
            area: 1.0,
        };
        let interfaces = HeatTransferCalc::interface_temperature(&[l1, l2], 100.0, 0.0);
        assert_eq!(interfaces.len(), 1);
        // By symmetry, mid-temperature = 50 °C
        assert!(approx_eq(interfaces[0], 50.0, 1e-9));
    }

    #[test]
    fn test_interface_temperature_three_layers() {
        let l1 = ThermalLayer {
            material: Material::concrete(),
            thickness: 0.1,
            area: 1.0,
        }; // R = 0.1
        let l2 = ThermalLayer {
            material: Material::wood(),
            thickness: 0.12,
            area: 1.0,
        }; // R = 0.12/0.12 = 1.0
        let l3 = ThermalLayer {
            material: Material::concrete(),
            thickness: 0.1,
            area: 1.0,
        }; // R = 0.1
        let interfaces = HeatTransferCalc::interface_temperature(&[l1, l2, l3], 100.0, 0.0);
        assert_eq!(interfaces.len(), 2);
        // Total R = 0.1 + 1.0 + 0.1 = 1.2; Q = 100/1.2 ≈ 83.33 W
        // T1 = 100 - 83.33*0.1 ≈ 91.67
        // T2 = 91.67 - 83.33*1.0 ≈ 8.33
        let q = 100.0 / 1.2;
        assert!(approx_eq(interfaces[0], 100.0 - q * 0.1, 1e-6));
        assert!(approx_eq(interfaces[1], interfaces[0] - q * 1.0, 1e-6));
    }

    #[test]
    fn test_interface_temperature_monotone() {
        let layers = vec![
            ThermalLayer {
                material: Material::steel(),
                thickness: 0.01,
                area: 1.0,
            },
            ThermalLayer {
                material: Material::concrete(),
                thickness: 0.2,
                area: 1.0,
            },
            ThermalLayer {
                material: Material::wood(),
                thickness: 0.05,
                area: 1.0,
            },
        ];
        let temps = HeatTransferCalc::interface_temperature(&layers, 200.0, 10.0);
        assert_eq!(temps.len(), 2);
        // Must be monotonically decreasing
        assert!(temps[0] > temps[1]);
        assert!(temps[0] < 200.0);
        assert!(temps[1] > 10.0);
    }

    // --- HeatTransferMode enum ---

    #[test]
    fn test_mode_variants() {
        let modes = [
            HeatTransferMode::Conduction,
            HeatTransferMode::Convection,
            HeatTransferMode::Radiation,
            HeatTransferMode::Combined,
        ];
        for m in &modes {
            let cloned = m.clone();
            assert_eq!(&cloned, m);
        }
    }

    #[test]
    fn test_mode_debug() {
        let s = format!("{:?}", HeatTransferMode::Conduction);
        assert_eq!(s, "Conduction");
    }

    // --- Additional edge-case tests ---

    #[test]
    fn test_parallel_resistance_single_layer() {
        let l = ThermalLayer {
            material: Material::concrete(),
            thickness: 0.1,
            area: 1.0,
        };
        let r_par = HeatTransferCalc::parallel_resistance(std::slice::from_ref(&l));
        let r_ser = HeatTransferCalc::series_resistance(std::slice::from_ref(&l));
        assert!(approx_eq(r_par, r_ser, EPS));
    }

    #[test]
    fn test_conduction_flux_symmetry() {
        let flux_ab =
            HeatTransferCalc::conduction_heat_flux(&Material::copper(), 100.0, 20.0, 0.01);
        let flux_ba =
            HeatTransferCalc::conduction_heat_flux(&Material::copper(), 20.0, 100.0, 0.01);
        assert!(approx_eq(flux_ab, -flux_ba, EPS));
    }

    #[test]
    fn test_radiation_stefan_boltzmann_value() {
        assert!((STEFAN_BOLTZMANN - 5.67e-8).abs() < 1e-10);
    }

    #[test]
    fn test_heat_flow_conservation_series() {
        // Q_total must equal Q through each individual layer (series law)
        let l1 = ThermalLayer {
            material: Material::steel(),
            thickness: 0.005,
            area: 1.0,
        };
        let l2 = ThermalLayer {
            material: Material::concrete(),
            thickness: 0.15,
            area: 1.0,
        };
        let q_total = HeatTransferCalc::heat_flow_series(&[l1.clone(), l2.clone()], 300.0, 20.0);
        let r_total = HeatTransferCalc::series_resistance(&[l1, l2]);
        assert!(approx_eq(q_total, (300.0 - 20.0) / r_total, 1e-9));
    }
}
