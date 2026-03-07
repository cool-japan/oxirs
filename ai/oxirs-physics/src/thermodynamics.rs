//! # Thermodynamics Module
//!
//! Thermodynamic computations for ideal gases, entropy, enthalpy, Carnot efficiency,
//! adiabatic processes, Clausius-Clapeyron equation, and heat transfer.
//!
//! ## Physical Constants
//!
//! - Universal gas constant: `R = 8.314462618 J/(mol·K)`
//!
//! ## Examples
//!
//! ```rust
//! use oxirs_physics::thermodynamics::{ThermodynamicsCalculator, GasProperties};
//!
//! // Ideal gas pressure
//! let p = ThermodynamicsCalculator::ideal_gas_pressure(1.0, 273.15, 22.4e-3);
//! assert!((p - 101_325.0).abs() < 2000.0);  // approximately 1 atm
//!
//! // Carnot efficiency
//! let eta = ThermodynamicsCalculator::carnot_efficiency(600.0, 300.0);
//! assert!((eta - 0.5).abs() < 1e-10);
//! ```

use std::f64::consts::E;

/// Thermodynamic state of a gas sample
#[derive(Debug, Clone, PartialEq)]
pub struct ThermodynamicState {
    /// Temperature in Kelvin
    pub temperature_k: f64,
    /// Pressure in Pascals
    pub pressure_pa: f64,
    /// Volume in cubic metres
    pub volume_m3: f64,
    /// Amount of substance in moles
    pub moles: f64,
}

/// Thermophysical properties of a gas species
#[derive(Debug, Clone, PartialEq)]
pub struct GasProperties {
    /// Molar heat capacity at constant pressure [J/(mol·K)]
    pub cp: f64,
    /// Molar heat capacity at constant volume [J/(mol·K)]
    pub cv: f64,
    /// Heat capacity ratio: γ = Cp / Cv  (dimensionless)
    pub gamma: f64,
    /// Molar mass [kg/mol]
    pub molar_mass: f64,
}

impl GasProperties {
    /// Diatomic ideal gas (N₂, O₂, air-like): γ = 7/5 = 1.4
    ///
    /// Cp = 7R/2, Cv = 5R/2 per the equipartition theorem.
    pub fn ideal_diatomic() -> Self {
        let r = ThermodynamicsCalculator::R;
        let cv = 2.5 * r;
        let cp = 3.5 * r;
        Self {
            cp,
            cv,
            gamma: cp / cv,
            molar_mass: 0.029, // ≈ air 29 g/mol
        }
    }

    /// Monatomic ideal gas (noble gases He, Ar, Ne): γ = 5/3
    ///
    /// Cp = 5R/2, Cv = 3R/2 per the equipartition theorem.
    pub fn ideal_monatomic() -> Self {
        let r = ThermodynamicsCalculator::R;
        let cv = 1.5 * r;
        let cp = 2.5 * r;
        Self {
            cp,
            cv,
            gamma: cp / cv,
            molar_mass: 0.040, // argon 40 g/mol
        }
    }

    /// Approximate properties of dry air (γ = 1.4, Cp ≈ 1005 J/(kg·K))
    ///
    /// Molar values computed from the specific heat and molar mass of air.
    pub fn air() -> Self {
        let molar_mass = 0.028_966; // kg/mol
        let cp = 1005.0 * molar_mass; // J/(mol·K)
        let r = ThermodynamicsCalculator::R;
        let cv = cp - r;
        Self {
            cp,
            cv,
            gamma: cp / cv,
            molar_mass,
        }
    }
}

/// Stateless calculator for thermodynamic relations
pub struct ThermodynamicsCalculator;

impl ThermodynamicsCalculator {
    /// Universal gas constant R = 8.314 462 618 J/(mol·K)
    pub const R: f64 = 8.314_462_618;

    // ─── Ideal gas law: PV = nRT ─────────────────────────────────────────────

    /// Compute pressure from ideal gas law: P = nRT / V
    ///
    /// # Arguments
    /// * `n`    – amount of substance [mol]
    /// * `t_k`  – temperature [K]
    /// * `v_m3` – volume [m³]
    pub fn ideal_gas_pressure(n: f64, t_k: f64, v_m3: f64) -> f64 {
        n * Self::R * t_k / v_m3
    }

    /// Compute volume from ideal gas law: V = nRT / P
    ///
    /// # Arguments
    /// * `n`   – amount of substance [mol]
    /// * `t_k` – temperature [K]
    /// * `p_pa` – pressure [Pa]
    pub fn ideal_gas_volume(n: f64, t_k: f64, p_pa: f64) -> f64 {
        n * Self::R * t_k / p_pa
    }

    /// Compute temperature from ideal gas law: T = PV / (nR)
    ///
    /// # Arguments
    /// * `n`    – amount of substance [mol]
    /// * `p_pa` – pressure [Pa]
    /// * `v_m3` – volume [m³]
    pub fn ideal_gas_temperature(n: f64, p_pa: f64, v_m3: f64) -> f64 {
        p_pa * v_m3 / (n * Self::R)
    }

    // ─── Entropy changes ─────────────────────────────────────────────────────

    /// Entropy change for an isothermal process: ΔS = nR·ln(V₂/V₁)
    ///
    /// Positive when the gas expands (V₂ > V₁).
    ///
    /// # Arguments
    /// * `n`  – amount of substance [mol]
    /// * `v1` – initial volume [m³]
    /// * `v2` – final volume [m³]
    pub fn entropy_change_isothermal(n: f64, v1: f64, v2: f64) -> f64 {
        n * Self::R * (v2 / v1).ln()
    }

    /// Entropy change for an isobaric process: ΔS = n·Cp·ln(T₂/T₁)
    ///
    /// # Arguments
    /// * `n`  – amount of substance [mol]
    /// * `cp` – molar heat capacity at constant pressure [J/(mol·K)]
    /// * `t1` – initial temperature [K]
    /// * `t2` – final temperature [K]
    pub fn entropy_change_isobaric(n: f64, cp: f64, t1: f64, t2: f64) -> f64 {
        n * cp * (t2 / t1).ln()
    }

    // ─── Heat engine efficiency ───────────────────────────────────────────────

    /// Carnot efficiency: η = 1 − T_cold / T_hot
    ///
    /// Both temperatures must be in Kelvin; returns a value in [0, 1).
    ///
    /// # Arguments
    /// * `t_hot_k`  – hot reservoir temperature [K]
    /// * `t_cold_k` – cold reservoir temperature [K]
    pub fn carnot_efficiency(t_hot_k: f64, t_cold_k: f64) -> f64 {
        1.0 - t_cold_k / t_hot_k
    }

    // ─── Adiabatic process ────────────────────────────────────────────────────

    /// Temperature after an adiabatic (reversible, no heat exchange) expansion/compression.
    ///
    /// T₂ = T₁ · (V₁ / V₂)^(γ − 1)
    ///
    /// # Arguments
    /// * `t1_k`  – initial temperature [K]
    /// * `v1`    – initial volume [m³]
    /// * `v2`    – final volume [m³]
    /// * `gamma` – heat capacity ratio Cp/Cv (dimensionless)
    pub fn adiabatic_temperature(t1_k: f64, v1: f64, v2: f64, gamma: f64) -> f64 {
        t1_k * (v1 / v2).powf(gamma - 1.0)
    }

    // ─── Enthalpy ─────────────────────────────────────────────────────────────

    /// Enthalpy change at constant pressure: ΔH = n·Cp·ΔT
    ///
    /// # Arguments
    /// * `n`    – amount of substance [mol]
    /// * `cp`   – molar heat capacity at constant pressure [J/(mol·K)]
    /// * `t1_k` – initial temperature [K]
    /// * `t2_k` – final temperature [K]
    pub fn enthalpy_change(n: f64, cp: f64, t1_k: f64, t2_k: f64) -> f64 {
        n * cp * (t2_k - t1_k)
    }

    // ─── Heat capacity ratio ──────────────────────────────────────────────────

    /// Heat capacity ratio: γ = Cp / Cv
    pub fn gamma(cp: f64, cv: f64) -> f64 {
        cp / cv
    }

    // ─── Fourier heat conduction ──────────────────────────────────────────────

    /// Steady-state conduction heat transfer rate (Fourier's law):
    /// Q/t = k · A · ΔT / L  [W]
    ///
    /// # Arguments
    /// * `k`       – thermal conductivity [W/(m·K)]
    /// * `area`    – cross-sectional area [m²]
    /// * `delta_t` – temperature difference across the material [K or °C]
    /// * `thickness` – material thickness L [m]
    pub fn heat_transfer_rate(k: f64, area: f64, delta_t: f64, thickness: f64) -> f64 {
        k * area * delta_t / thickness
    }

    // ─── Phase transitions ────────────────────────────────────────────────────

    /// Clausius-Clapeyron equation — predict saturation pressure at T₂:
    ///
    /// ln(P₂/P₁) = (L/R) · (1/T₁ − 1/T₂)
    ///
    /// # Arguments
    /// * `p1`           – saturation pressure at T₁ [Pa]
    /// * `t1_k`         – reference temperature [K]
    /// * `t2_k`         – target temperature [K]
    /// * `latent_heat`  – specific latent heat of vaporisation [J/mol]
    ///
    /// Returns P₂ in Pascals.
    pub fn clausius_clapeyron(p1: f64, t1_k: f64, t2_k: f64, latent_heat: f64) -> f64 {
        let exponent = (latent_heat / Self::R) * (1.0 / t1_k - 1.0 / t2_k);
        p1 * E.powf(exponent)
    }

    // ─── Temperature unit conversion ──────────────────────────────────────────

    /// Convert degrees Celsius to Kelvin: T(K) = T(°C) + 273.15
    pub fn celsius_to_kelvin(c: f64) -> f64 {
        c + 273.15
    }

    /// Convert Kelvin to degrees Celsius: T(°C) = T(K) − 273.15
    pub fn kelvin_to_celsius(k: f64) -> f64 {
        k - 273.15
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const R: f64 = ThermodynamicsCalculator::R;
    const EPS: f64 = 1e-9;

    // ── Ideal gas law ──────────────────────────────────────────────────────

    #[test]
    fn test_ideal_gas_pressure_basic() {
        // 1 mol, 300 K, 1 m³ → P = R * 300 ≈ 2494 Pa
        let p = ThermodynamicsCalculator::ideal_gas_pressure(1.0, 300.0, 1.0);
        assert!((p - R * 300.0).abs() < EPS);
    }

    #[test]
    fn test_ideal_gas_pressure_one_atm() {
        // 1 mol at STP: T = 273.15 K, V ≈ 22.414 L = 0.022414 m³
        let p = ThermodynamicsCalculator::ideal_gas_pressure(1.0, 273.15, 0.022_414);
        assert!((p - 101_325.0).abs() < 500.0, "pressure = {}", p);
    }

    #[test]
    fn test_ideal_gas_pressure_two_moles() {
        let p = ThermodynamicsCalculator::ideal_gas_pressure(2.0, 300.0, 1.0);
        assert!((p - 2.0 * R * 300.0).abs() < EPS);
    }

    #[test]
    fn test_ideal_gas_volume_basic() {
        let v = ThermodynamicsCalculator::ideal_gas_volume(1.0, 300.0, R * 300.0);
        assert!((v - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_ideal_gas_volume_stp() {
        // 1 mol at STP P = 101325 Pa, T = 273.15 K → V ≈ 22.414 L
        let v = ThermodynamicsCalculator::ideal_gas_volume(1.0, 273.15, 101_325.0);
        assert!((v - 0.022_414).abs() < 5e-5, "volume = {}", v);
    }

    #[test]
    fn test_ideal_gas_volume_proportional_to_moles() {
        let v1 = ThermodynamicsCalculator::ideal_gas_volume(1.0, 300.0, 100_000.0);
        let v2 = ThermodynamicsCalculator::ideal_gas_volume(3.0, 300.0, 100_000.0);
        assert!((v2 - 3.0 * v1).abs() < 1e-12);
    }

    #[test]
    fn test_ideal_gas_temperature_basic() {
        // T = PV / (nR).  Use P = nRT/V = R * 500 when n=1, V=1
        let t = ThermodynamicsCalculator::ideal_gas_temperature(1.0, R * 500.0, 1.0);
        assert!((t - 500.0).abs() < EPS);
    }

    #[test]
    fn test_ideal_gas_roundtrip_p_v_t() {
        let n = 2.0;
        let t = 400.0;
        let v = 0.5;
        let p = ThermodynamicsCalculator::ideal_gas_pressure(n, t, v);
        let v2 = ThermodynamicsCalculator::ideal_gas_volume(n, t, p);
        assert!((v2 - v).abs() < 1e-12);
        let t2 = ThermodynamicsCalculator::ideal_gas_temperature(n, p, v);
        assert!((t2 - t).abs() < 1e-9);
    }

    // ── Entropy ────────────────────────────────────────────────────────────

    #[test]
    fn test_entropy_change_isothermal_expansion_positive() {
        // Expansion: V2 > V1 → ΔS > 0
        let ds = ThermodynamicsCalculator::entropy_change_isothermal(1.0, 1.0, 2.0);
        assert!(ds > 0.0);
    }

    #[test]
    fn test_entropy_change_isothermal_compression_negative() {
        let ds = ThermodynamicsCalculator::entropy_change_isothermal(1.0, 2.0, 1.0);
        assert!(ds < 0.0);
    }

    #[test]
    fn test_entropy_change_isothermal_formula() {
        // ΔS = nR ln(V2/V1)
        let n = 2.0;
        let v1 = 1.0;
        let v2 = std::f64::consts::E; // ln(e) = 1
        let ds = ThermodynamicsCalculator::entropy_change_isothermal(n, v1, v2);
        assert!((ds - n * R).abs() < 1e-10);
    }

    #[test]
    fn test_entropy_change_isothermal_same_volume() {
        let ds = ThermodynamicsCalculator::entropy_change_isothermal(1.0, 1.5, 1.5);
        assert!(ds.abs() < EPS);
    }

    #[test]
    fn test_entropy_change_isobaric_heating_positive() {
        let cp = 29.1; // diatomic J/(mol·K)
        let ds = ThermodynamicsCalculator::entropy_change_isobaric(1.0, cp, 300.0, 600.0);
        assert!(ds > 0.0);
    }

    #[test]
    fn test_entropy_change_isobaric_cooling_negative() {
        let cp = 29.1;
        let ds = ThermodynamicsCalculator::entropy_change_isobaric(1.0, cp, 600.0, 300.0);
        assert!(ds < 0.0);
    }

    #[test]
    fn test_entropy_change_isobaric_formula() {
        // ΔS = n·Cp·ln(T2/T1); with T2 = e·T1 → ΔS = n·Cp
        let n = 1.0;
        let cp = 10.0;
        let t1 = 100.0;
        let t2 = t1 * std::f64::consts::E;
        let ds = ThermodynamicsCalculator::entropy_change_isobaric(n, cp, t1, t2);
        assert!((ds - n * cp).abs() < 1e-10);
    }

    // ── Carnot efficiency ──────────────────────────────────────────────────

    #[test]
    fn test_carnot_efficiency_half() {
        let eta = ThermodynamicsCalculator::carnot_efficiency(600.0, 300.0);
        assert!((eta - 0.5).abs() < EPS);
    }

    #[test]
    fn test_carnot_efficiency_range_0_to_1() {
        let eta = ThermodynamicsCalculator::carnot_efficiency(1000.0, 200.0);
        assert!(eta > 0.0 && eta < 1.0);
    }

    #[test]
    fn test_carnot_efficiency_equal_temps_zero() {
        let eta = ThermodynamicsCalculator::carnot_efficiency(300.0, 300.0);
        assert!(eta.abs() < EPS);
    }

    #[test]
    fn test_carnot_efficiency_high_temp() {
        // T_cold → 0 ⟹ η → 1
        let eta = ThermodynamicsCalculator::carnot_efficiency(1000.0, 1.0);
        assert!(eta > 0.99);
    }

    #[test]
    fn test_carnot_efficiency_formula() {
        let t_hot = 800.0;
        let t_cold = 200.0;
        let expected = 1.0 - t_cold / t_hot;
        let eta = ThermodynamicsCalculator::carnot_efficiency(t_hot, t_cold);
        assert!((eta - expected).abs() < EPS);
    }

    // ── Adiabatic process ──────────────────────────────────────────────────

    #[test]
    fn test_adiabatic_temperature_expansion_cools() {
        // Expanding gas cools: V2 > V1 → T2 < T1
        let t2 = ThermodynamicsCalculator::adiabatic_temperature(300.0, 1.0, 2.0, 1.4);
        assert!(t2 < 300.0);
    }

    #[test]
    fn test_adiabatic_temperature_compression_heats() {
        let t2 = ThermodynamicsCalculator::adiabatic_temperature(300.0, 2.0, 1.0, 1.4);
        assert!(t2 > 300.0);
    }

    #[test]
    fn test_adiabatic_temperature_same_volume() {
        let t2 = ThermodynamicsCalculator::adiabatic_temperature(300.0, 1.0, 1.0, 1.4);
        assert!((t2 - 300.0).abs() < EPS);
    }

    #[test]
    fn test_adiabatic_temperature_formula() {
        // T2 = T1 * (V1/V2)^(γ-1)
        let t1: f64 = 500.0;
        let v1: f64 = 2.0;
        let v2: f64 = 4.0;
        let gamma: f64 = 1.4;
        let expected = t1 * (v1 / v2).powf(gamma - 1.0);
        let t2 = ThermodynamicsCalculator::adiabatic_temperature(t1, v1, v2, gamma);
        assert!((t2 - expected).abs() < 1e-10);
    }

    #[test]
    fn test_adiabatic_temperature_monatomic_gamma() {
        // γ = 5/3 for monatomic gas
        let t2 = ThermodynamicsCalculator::adiabatic_temperature(300.0, 1.0, 8.0, 5.0 / 3.0);
        let expected = 300.0 * (1.0_f64 / 8.0).powf(2.0 / 3.0);
        assert!((t2 - expected).abs() < 1e-9);
    }

    // ── Enthalpy ───────────────────────────────────────────────────────────

    #[test]
    fn test_enthalpy_change_positive() {
        let dh = ThermodynamicsCalculator::enthalpy_change(1.0, 29.1, 300.0, 400.0);
        assert!(dh > 0.0);
    }

    #[test]
    fn test_enthalpy_change_negative_cooling() {
        let dh = ThermodynamicsCalculator::enthalpy_change(1.0, 29.1, 400.0, 300.0);
        assert!(dh < 0.0);
    }

    #[test]
    fn test_enthalpy_change_formula() {
        let n = 2.0;
        let cp = 30.0;
        let t1 = 200.0;
        let t2 = 350.0;
        let expected = n * cp * (t2 - t1);
        let dh = ThermodynamicsCalculator::enthalpy_change(n, cp, t1, t2);
        assert!((dh - expected).abs() < EPS);
    }

    #[test]
    fn test_enthalpy_change_zero_delta_t() {
        let dh = ThermodynamicsCalculator::enthalpy_change(1.0, 29.1, 300.0, 300.0);
        assert!(dh.abs() < EPS);
    }

    // ── Gamma ──────────────────────────────────────────────────────────────

    #[test]
    fn test_gamma_diatomic() {
        let cp = 3.5 * R;
        let cv = 2.5 * R;
        let g = ThermodynamicsCalculator::gamma(cp, cv);
        assert!((g - 1.4).abs() < 1e-10);
    }

    #[test]
    fn test_gamma_monatomic() {
        let cp = 2.5 * R;
        let cv = 1.5 * R;
        let g = ThermodynamicsCalculator::gamma(cp, cv);
        assert!((g - 5.0 / 3.0).abs() < 1e-10);
    }

    // ── Heat transfer rate ──────────────────────────────────────────────────

    #[test]
    fn test_heat_transfer_rate_basic() {
        // Q/t = k * A * ΔT / L
        let q = ThermodynamicsCalculator::heat_transfer_rate(1.0, 1.0, 10.0, 1.0);
        assert!((q - 10.0).abs() < EPS);
    }

    #[test]
    fn test_heat_transfer_rate_glass_window() {
        // Glass: k ≈ 1 W/(m·K), A = 1 m², ΔT = 20 K, L = 0.01 m → Q = 2000 W
        let q = ThermodynamicsCalculator::heat_transfer_rate(1.0, 1.0, 20.0, 0.01);
        assert!((q - 2000.0).abs() < EPS);
    }

    #[test]
    fn test_heat_transfer_rate_thicker_less() {
        let q1 = ThermodynamicsCalculator::heat_transfer_rate(0.5, 2.0, 30.0, 0.1);
        let q2 = ThermodynamicsCalculator::heat_transfer_rate(0.5, 2.0, 30.0, 0.2);
        assert!(q1 > q2);
    }

    #[test]
    fn test_heat_transfer_rate_proportional_to_conductivity() {
        let q1 = ThermodynamicsCalculator::heat_transfer_rate(1.0, 1.0, 10.0, 1.0);
        let q2 = ThermodynamicsCalculator::heat_transfer_rate(2.0, 1.0, 10.0, 1.0);
        assert!((q2 - 2.0 * q1).abs() < EPS);
    }

    // ── Clausius-Clapeyron ─────────────────────────────────────────────────

    #[test]
    fn test_clausius_clapeyron_water_boiling() {
        // Water: L ≈ 40_700 J/mol, at 373.15 K P1 = 101325 Pa
        // At 383 K (10 K above boiling) pressure should be higher
        let p2 = ThermodynamicsCalculator::clausius_clapeyron(101_325.0, 373.15, 383.15, 40_700.0);
        assert!(p2 > 101_325.0, "p2 = {}", p2);
    }

    #[test]
    fn test_clausius_clapeyron_same_temperature() {
        let p2 = ThermodynamicsCalculator::clausius_clapeyron(101_325.0, 373.15, 373.15, 40_700.0);
        assert!((p2 - 101_325.0).abs() < EPS);
    }

    #[test]
    fn test_clausius_clapeyron_lower_temperature() {
        // Cooling below boiling point → lower saturation pressure
        let p2 = ThermodynamicsCalculator::clausius_clapeyron(101_325.0, 373.15, 363.15, 40_700.0);
        assert!(p2 < 101_325.0, "p2 = {}", p2);
    }

    // ── Temperature conversion ─────────────────────────────────────────────

    #[test]
    fn test_celsius_to_kelvin_zero() {
        let k = ThermodynamicsCalculator::celsius_to_kelvin(0.0);
        assert!((k - 273.15).abs() < EPS);
    }

    #[test]
    fn test_celsius_to_kelvin_boiling_water() {
        let k = ThermodynamicsCalculator::celsius_to_kelvin(100.0);
        assert!((k - 373.15).abs() < EPS);
    }

    #[test]
    fn test_celsius_to_kelvin_negative() {
        let k = ThermodynamicsCalculator::celsius_to_kelvin(-40.0);
        assert!((k - 233.15).abs() < EPS);
    }

    #[test]
    fn test_kelvin_to_celsius_absolute_zero() {
        let c = ThermodynamicsCalculator::kelvin_to_celsius(0.0);
        assert!((c - (-273.15)).abs() < EPS);
    }

    #[test]
    fn test_kelvin_to_celsius_body_temperature() {
        let c = ThermodynamicsCalculator::kelvin_to_celsius(310.15);
        assert!((c - 37.0).abs() < EPS);
    }

    #[test]
    fn test_celsius_kelvin_roundtrip() {
        let c = 42.0;
        let k = ThermodynamicsCalculator::celsius_to_kelvin(c);
        let c2 = ThermodynamicsCalculator::kelvin_to_celsius(k);
        assert!((c2 - c).abs() < EPS);
    }

    // ── GasProperties ──────────────────────────────────────────────────────

    #[test]
    fn test_gas_properties_diatomic_gamma() {
        let props = GasProperties::ideal_diatomic();
        assert!((props.gamma - 1.4).abs() < 1e-10);
    }

    #[test]
    fn test_gas_properties_monatomic_gamma() {
        let props = GasProperties::ideal_monatomic();
        assert!((props.gamma - 5.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_gas_properties_diatomic_cp_cv_relation() {
        let props = GasProperties::ideal_diatomic();
        // Cp - Cv = R  (Mayer's relation for ideal gas)
        assert!((props.cp - props.cv - R).abs() < 1e-9);
    }

    #[test]
    fn test_gas_properties_monatomic_cp_cv_relation() {
        let props = GasProperties::ideal_monatomic();
        assert!((props.cp - props.cv - R).abs() < 1e-9);
    }

    #[test]
    fn test_gas_properties_air_gamma_approx() {
        let props = GasProperties::air();
        // Air γ ≈ 1.4 within 1%
        assert!((props.gamma - 1.4).abs() < 0.01, "gamma = {}", props.gamma);
    }

    #[test]
    fn test_gas_properties_air_molar_mass() {
        let props = GasProperties::air();
        assert!((props.molar_mass - 0.028_966).abs() < 1e-6);
    }

    #[test]
    fn test_gas_properties_diatomic_cv_positive() {
        let props = GasProperties::ideal_diatomic();
        assert!(props.cv > 0.0);
        assert!(props.cp > props.cv);
    }

    // ── Extra coverage ─────────────────────────────────────────────────────

    #[test]
    fn test_ideal_gas_pressure_high_temperature() {
        let p = ThermodynamicsCalculator::ideal_gas_pressure(5.0, 1000.0, 0.1);
        assert!(p > 0.0);
        assert!((p - 5.0 * R * 1000.0 / 0.1).abs() < 1e-5);
    }

    #[test]
    fn test_entropy_change_double_moles() {
        let ds1 = ThermodynamicsCalculator::entropy_change_isothermal(1.0, 1.0, 3.0);
        let ds2 = ThermodynamicsCalculator::entropy_change_isothermal(2.0, 1.0, 3.0);
        assert!((ds2 - 2.0 * ds1).abs() < 1e-12);
    }

    #[test]
    fn test_heat_transfer_rate_zero_delta_t() {
        let q = ThermodynamicsCalculator::heat_transfer_rate(50.0, 1.0, 0.0, 0.01);
        assert!(q.abs() < EPS);
    }
}
