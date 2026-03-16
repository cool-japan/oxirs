//! # Statistical Mechanics Module
//!
//! Classical and quantum statistical mechanics computations:
//!
//! - **Maxwell-Boltzmann speed distribution** — probability density and CDF
//!   for ideal gas molecules.
//! - **Partition function** — canonical ensemble over discrete energy levels.
//! - **Boltzmann factor** — `exp(-E / k_B T)` with normalisation.
//! - **Thermodynamic entropy from microstates** — `S = k_B ln Ω`.
//! - **Equipartition theorem** — average energy per degree of freedom.
//! - **Fermi-Dirac distribution** — occupation of fermionic energy levels.
//! - **Bose-Einstein distribution** — occupation of bosonic energy levels.
//! - **Mean free path** — ideal gas kinetic theory.
//!
//! ## Physical Constants
//!
//! | Symbol | Value | Unit |
//! |--------|-------|------|
//! | k_B | 1.380649 × 10⁻²³ | J/K |
//! | N_A | 6.02214076 × 10²³ | mol⁻¹ |
//!
//! ## Example
//!
//! ```rust
//! use oxirs_physics::statistical_mechanics::{
//!     StatMech, MaxwellBoltzmannDist,
//! };
//!
//! // Maxwell-Boltzmann speed distribution for N₂ at 300 K
//! let molar_mass_n2 = 0.028; // kg/mol
//! let mass_n2 = molar_mass_n2 / StatMech::N_A;
//! let dist = MaxwellBoltzmannDist::new(mass_n2, 300.0);
//!
//! // Most probable speed should be ~422 m/s for N₂ at 300 K
//! let v_p = dist.most_probable_speed();
//! assert!((v_p - 422.0).abs() < 5.0);
//! ```

use std::f64::consts::PI;

// ─────────────────────────────────────────────────────────────────────────────
// Physical constants
// ─────────────────────────────────────────────────────────────────────────────

/// Core statistical mechanics calculator — holds physical constants and
/// provides static helper methods.
pub struct StatMech;

impl StatMech {
    /// Boltzmann constant [J/K].
    pub const K_B: f64 = 1.380_649e-23;

    /// Avogadro's number [mol⁻¹].
    pub const N_A: f64 = 6.022_140_76e23;

    /// Planck constant [J·s].
    pub const H: f64 = 6.626_070_15e-34;

    // ── Boltzmann factor ──────────────────────────────────────────────────────

    /// Boltzmann factor: `exp(-E / (k_B · T))`.
    ///
    /// * `energy_j` — energy level in Joules.
    /// * `temperature_k` — temperature in Kelvin (must be > 0).
    pub fn boltzmann_factor(energy_j: f64, temperature_k: f64) -> f64 {
        if temperature_k <= 0.0 {
            return 0.0;
        }
        (-energy_j / (Self::K_B * temperature_k)).exp()
    }

    // ── Partition function ─────────────────────────────────────────────────────

    /// Canonical partition function for a system with discrete energy levels.
    ///
    /// `Z = Σ g_i · exp(-E_i / k_B T)`
    ///
    /// * `levels` — slice of `(energy_J, degeneracy)` pairs.
    /// * `temperature_k` — temperature in Kelvin.
    pub fn partition_function(levels: &[(f64, f64)], temperature_k: f64) -> f64 {
        if temperature_k <= 0.0 {
            return 0.0;
        }
        levels
            .iter()
            .map(|(e, g)| g * Self::boltzmann_factor(*e, temperature_k))
            .sum()
    }

    /// Normalised Boltzmann probabilities for a set of energy levels.
    ///
    /// Returns a vector of probabilities `p_i = g_i · exp(-E_i/kT) / Z`.
    pub fn boltzmann_probabilities(levels: &[(f64, f64)], temperature_k: f64) -> Vec<f64> {
        let z = Self::partition_function(levels, temperature_k);
        if z < f64::EPSILON {
            return vec![0.0; levels.len()];
        }
        levels
            .iter()
            .map(|(e, g)| g * Self::boltzmann_factor(*e, temperature_k) / z)
            .collect()
    }

    // ── Entropy ───────────────────────────────────────────────────────────────

    /// Boltzmann entropy from the number of microstates: `S = k_B · ln(Ω)`.
    ///
    /// * `microstates` — Ω, the number of accessible microstates (must be ≥ 1).
    pub fn entropy_from_microstates(microstates: f64) -> f64 {
        if microstates < 1.0 {
            return 0.0;
        }
        Self::K_B * microstates.ln()
    }

    /// Gibbs entropy: `S = -k_B · Σ p_i ln(p_i)` for a probability
    /// distribution over microstates.
    pub fn gibbs_entropy(probabilities: &[f64]) -> f64 {
        -Self::K_B
            * probabilities
                .iter()
                .filter(|&&p| p > 0.0)
                .map(|&p| p * p.ln())
                .sum::<f64>()
    }

    // ── Equipartition theorem ─────────────────────────────────────────────────

    /// Average energy per degree of freedom: `<E> = k_B T / 2`.
    ///
    /// * `temperature_k` — temperature in Kelvin.
    pub fn energy_per_dof(temperature_k: f64) -> f64 {
        0.5 * Self::K_B * temperature_k
    }

    /// Total average energy for `f` degrees of freedom: `<E> = f · k_B T / 2`.
    pub fn total_average_energy(degrees_of_freedom: u32, temperature_k: f64) -> f64 {
        degrees_of_freedom as f64 * Self::energy_per_dof(temperature_k)
    }

    // ── Fermi-Dirac distribution ──────────────────────────────────────────────

    /// Fermi-Dirac occupation number.
    ///
    /// `f(E) = 1 / (exp((E - μ) / k_B T) + 1)`
    ///
    /// * `energy_j` — energy of the state in Joules.
    /// * `chemical_potential_j` — Fermi level / chemical potential μ \[J\].
    /// * `temperature_k` — temperature \[K\].  When `T → 0` the result
    ///   approaches a step function.
    pub fn fermi_dirac(energy_j: f64, chemical_potential_j: f64, temperature_k: f64) -> f64 {
        if temperature_k <= 0.0 {
            // Zero-temperature limit: step function
            return if energy_j <= chemical_potential_j {
                1.0
            } else {
                0.0
            };
        }
        let exponent = (energy_j - chemical_potential_j) / (Self::K_B * temperature_k);
        1.0 / (exponent.exp() + 1.0)
    }

    // ── Bose-Einstein distribution ────────────────────────────────────────────

    /// Bose-Einstein occupation number.
    ///
    /// `n(E) = 1 / (exp((E - μ) / k_B T) - 1)`
    ///
    /// Returns `None` when the denominator would be zero or negative (invalid
    /// physical parameters).
    ///
    /// * `energy_j` — energy of the state \[J\].
    /// * `chemical_potential_j` — chemical potential μ \[J\] (must satisfy μ < E).
    /// * `temperature_k` — temperature \[K\].
    pub fn bose_einstein(
        energy_j: f64,
        chemical_potential_j: f64,
        temperature_k: f64,
    ) -> Option<f64> {
        if temperature_k <= 0.0 {
            return None;
        }
        let exponent = (energy_j - chemical_potential_j) / (Self::K_B * temperature_k);
        let denom = exponent.exp() - 1.0;
        if denom <= 0.0 {
            return None;
        }
        Some(1.0 / denom)
    }

    // ── Mean free path ────────────────────────────────────────────────────────

    /// Mean free path of an ideal gas molecule: `λ = 1 / (√2 · π · d² · n)`.
    ///
    /// * `diameter_m` — effective collision diameter \[m\].
    /// * `number_density` — molecules per m³.
    ///
    /// Returns `None` if inputs are non-positive.
    pub fn mean_free_path(diameter_m: f64, number_density: f64) -> Option<f64> {
        if diameter_m <= 0.0 || number_density <= 0.0 {
            return None;
        }
        let cross_section = PI * diameter_m * diameter_m;
        Some(1.0 / (2.0_f64.sqrt() * cross_section * number_density))
    }

    /// Mean free path using pressure and temperature (ideal gas law).
    ///
    /// Derived from `n = P / (k_B T)`:
    /// `λ = k_B T / (√2 · π · d² · P)`
    ///
    /// * `diameter_m` — collision diameter \[m\].
    /// * `pressure_pa` — gas pressure \[Pa\].
    /// * `temperature_k` — temperature \[K\].
    pub fn mean_free_path_from_pressure(
        diameter_m: f64,
        pressure_pa: f64,
        temperature_k: f64,
    ) -> Option<f64> {
        if diameter_m <= 0.0 || pressure_pa <= 0.0 || temperature_k <= 0.0 {
            return None;
        }
        let cross_section = PI * diameter_m * diameter_m;
        Some(Self::K_B * temperature_k / (2.0_f64.sqrt() * cross_section * pressure_pa))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Maxwell-Boltzmann speed distribution
// ─────────────────────────────────────────────────────────────────────────────

/// Maxwell-Boltzmann speed distribution for an ideal gas.
///
/// The probability density is:
/// `f(v) = 4π (m/(2π k_B T))^{3/2} v² exp(-m v²/(2 k_B T))`
#[derive(Debug, Clone)]
pub struct MaxwellBoltzmannDist {
    /// Particle mass \[kg\].
    pub mass_kg: f64,
    /// Temperature \[K\].
    pub temperature_k: f64,
}

impl MaxwellBoltzmannDist {
    /// Create a new Maxwell-Boltzmann distribution.
    ///
    /// # Panics
    ///
    /// Panics if `mass_kg ≤ 0` or `temperature_k ≤ 0`.
    pub fn new(mass_kg: f64, temperature_k: f64) -> Self {
        assert!(mass_kg > 0.0, "mass must be positive");
        assert!(temperature_k > 0.0, "temperature must be positive");
        Self {
            mass_kg,
            temperature_k,
        }
    }

    /// The characteristic thermal speed parameter: `a = sqrt(k_B T / m)`.
    pub fn thermal_speed(&self) -> f64 {
        (StatMech::K_B * self.temperature_k / self.mass_kg).sqrt()
    }

    /// Probability density at speed `v` [m/s].
    pub fn pdf(&self, v: f64) -> f64 {
        if v < 0.0 {
            return 0.0;
        }
        let a = self.thermal_speed();
        let a2 = a * a;
        // f(v) = 4π (1/(2π a²))^{3/2} v² exp(-v²/(2a²))
        //      = 4 / (√π · (2a²)^{3/2}) · v² exp(-v²/(2a²))
        let prefactor = 4.0
            * PI
            * (self.mass_kg / (2.0 * PI * StatMech::K_B * self.temperature_k))
                .powi(3)
                .sqrt();
        prefactor * v * v * (-v * v / (2.0 * a2)).exp()
    }

    /// Most probable speed: `v_p = √(2 k_B T / m)`.
    pub fn most_probable_speed(&self) -> f64 {
        let a = self.thermal_speed();
        2.0_f64.sqrt() * a
    }

    /// Mean speed: `<v> = √(8 k_B T / (π m))`.
    pub fn mean_speed(&self) -> f64 {
        let a = self.thermal_speed();
        (8.0 / PI).sqrt() * a
    }

    /// Root-mean-square speed: `v_rms = √(3 k_B T / m)`.
    pub fn rms_speed(&self) -> f64 {
        let a = self.thermal_speed();
        3.0_f64.sqrt() * a
    }

    /// Cumulative distribution function at speed `v` using numerical
    /// integration via the error function approximation.
    ///
    /// `CDF(v) = erf(x/√2) - √(2/π) · x · exp(-x²/2)`
    /// where `x = v / a`.
    pub fn cdf(&self, v: f64) -> f64 {
        if v <= 0.0 {
            return 0.0;
        }
        let a = self.thermal_speed();
        let x = v / (2.0_f64.sqrt() * a);
        // CDF = erf(v/(√(2)a)) - (2/π)^{1/2} · v/a · exp(-v²/(2a²))
        erf(x) - (2.0 / PI).sqrt() * (v / a) * (-(v * v) / (2.0 * a * a)).exp()
    }
}

/// Approximation of the error function using the Horner/rational polynomial
/// approximation (Abramowitz & Stegun 7.1.26, max error < 1.5 × 10⁻⁷).
fn erf(x: f64) -> f64 {
    if x < 0.0 {
        return -erf(-x);
    }
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    1.0 - poly * (-x * x).exp()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const K_B: f64 = StatMech::K_B;
    // Mass of N₂ molecule
    const MASS_N2: f64 = 0.028 / StatMech::N_A;

    // ── Boltzmann factor ──────────────────────────────────────────────────────

    #[test]
    fn test_boltzmann_factor_zero_energy() {
        let bf = StatMech::boltzmann_factor(0.0, 300.0);
        assert!((bf - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_boltzmann_factor_infinite_temperature() {
        // As T → ∞ the factor → 1 for finite E
        let bf = StatMech::boltzmann_factor(1e-20, 1e12);
        assert!((bf - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_boltzmann_factor_negative_temperature() {
        let bf = StatMech::boltzmann_factor(1e-21, -100.0);
        assert_eq!(bf, 0.0);
    }

    #[test]
    fn test_boltzmann_factor_large_energy() {
        // exp(-very large) → 0
        let bf = StatMech::boltzmann_factor(1.0e10, 1.0);
        assert!(bf < 1e-100);
    }

    // ── Partition function ─────────────────────────────────────────────────────

    #[test]
    fn test_partition_function_single_level() {
        let levels = [(0.0_f64, 1.0_f64)]; // ground state, g=1
        let z = StatMech::partition_function(&levels, 300.0);
        assert!((z - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_partition_function_degenerate_ground_state() {
        let levels = [(0.0_f64, 3.0_f64)]; // g=3
        let z = StatMech::partition_function(&levels, 300.0);
        assert!((z - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_partition_function_two_levels_low_temp() {
        let e1 = K_B * 1e6; // very high energy (k_B × 10^6 K)
        let levels = [(0.0_f64, 1.0_f64), (e1, 1.0_f64)];
        let z = StatMech::partition_function(&levels, 300.0);
        // Second level is essentially frozen out — Z ≈ 1
        assert!((z - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_partition_function_zero_temperature() {
        let levels = [(0.0_f64, 1.0_f64), (1e-20_f64, 1.0_f64)];
        let z = StatMech::partition_function(&levels, 0.0);
        assert_eq!(z, 0.0);
    }

    // ── Boltzmann probabilities ────────────────────────────────────────────────

    #[test]
    fn test_boltzmann_probabilities_sum_to_one() {
        let levels = [(0.0_f64, 1.0_f64), (K_B * 100.0, 1.0), (K_B * 200.0, 1.0)];
        let probs = StatMech::boltzmann_probabilities(&levels, 300.0);
        let total: f64 = probs.iter().sum();
        assert!((total - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_boltzmann_probabilities_ground_state_most_probable() {
        let levels = [
            (0.0_f64, 1.0_f64),
            (K_B * 1000.0, 1.0), // much higher energy
        ];
        let probs = StatMech::boltzmann_probabilities(&levels, 300.0);
        assert!(probs[0] > probs[1]);
    }

    #[test]
    fn test_boltzmann_probabilities_equal_when_same_energy() {
        let levels = [(0.0_f64, 1.0_f64), (0.0_f64, 1.0_f64)];
        let probs = StatMech::boltzmann_probabilities(&levels, 300.0);
        assert!((probs[0] - probs[1]).abs() < 1e-10);
    }

    // ── Entropy ───────────────────────────────────────────────────────────────

    #[test]
    fn test_entropy_one_microstate_is_zero() {
        let s = StatMech::entropy_from_microstates(1.0);
        assert!((s - 0.0).abs() < 1e-40);
    }

    #[test]
    fn test_entropy_increases_with_microstates() {
        let s1 = StatMech::entropy_from_microstates(10.0);
        let s2 = StatMech::entropy_from_microstates(100.0);
        assert!(s2 > s1);
    }

    #[test]
    fn test_entropy_formula_known_value() {
        let omega = 100.0_f64;
        let expected = K_B * omega.ln();
        let computed = StatMech::entropy_from_microstates(omega);
        assert!((computed - expected).abs() < 1e-40);
    }

    #[test]
    fn test_entropy_below_one_microstate() {
        let s = StatMech::entropy_from_microstates(0.5);
        assert_eq!(s, 0.0);
    }

    #[test]
    fn test_gibbs_entropy_uniform_distribution() {
        // For uniform distribution p_i = 1/N: S = k_B ln(N)
        let n = 4_usize;
        let probs = vec![0.25; n];
        let s = StatMech::gibbs_entropy(&probs);
        let expected = K_B * (n as f64).ln();
        assert!((s - expected).abs() < 1e-35);
    }

    #[test]
    fn test_gibbs_entropy_certain_state() {
        // Certain state: p1=1, rest=0 → S=0
        let probs = vec![1.0, 0.0, 0.0];
        let s = StatMech::gibbs_entropy(&probs);
        assert!(s.abs() < 1e-35);
    }

    // ── Equipartition ─────────────────────────────────────────────────────────

    #[test]
    fn test_energy_per_dof_300k() {
        let e = StatMech::energy_per_dof(300.0);
        let expected = 0.5 * K_B * 300.0;
        assert!((e - expected).abs() < 1e-35);
    }

    #[test]
    fn test_total_average_energy_diatomic() {
        // Diatomic ideal gas: 5 DOF → E = 5/2 k_B T
        let e = StatMech::total_average_energy(5, 300.0);
        let expected = 2.5 * K_B * 300.0;
        assert!((e - expected).abs() < 1e-35);
    }

    #[test]
    fn test_total_average_energy_zero_dof() {
        let e = StatMech::total_average_energy(0, 300.0);
        assert_eq!(e, 0.0);
    }

    // ── Fermi-Dirac ───────────────────────────────────────────────────────────

    #[test]
    fn test_fermi_dirac_below_fermi_level() {
        // E << μ → f ≈ 1
        let mu = 1e-19;
        let f = StatMech::fermi_dirac(1e-22, mu, 300.0);
        assert!(f > 0.99);
    }

    #[test]
    fn test_fermi_dirac_above_fermi_level() {
        // E >> μ → f ≈ 0
        let mu = 1e-22;
        let f = StatMech::fermi_dirac(1e-19, mu, 300.0);
        assert!(f < 0.01);
    }

    #[test]
    fn test_fermi_dirac_at_fermi_level() {
        // E = μ → f = 0.5 for any T > 0
        let mu = 1e-19;
        let f = StatMech::fermi_dirac(mu, mu, 300.0);
        assert!((f - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_fermi_dirac_zero_temperature_step() {
        let mu = 5e-20;
        let f_below = StatMech::fermi_dirac(1e-20, mu, 0.0);
        let f_above = StatMech::fermi_dirac(1e-19, mu, 0.0);
        assert_eq!(f_below, 1.0);
        assert_eq!(f_above, 0.0);
    }

    #[test]
    fn test_fermi_dirac_range() {
        for e_scale in [1e-21, 1e-20, 1e-19] {
            let f = StatMech::fermi_dirac(e_scale, 1e-20, 300.0);
            assert!((0.0..=1.0).contains(&f));
        }
    }

    // ── Bose-Einstein ─────────────────────────────────────────────────────────

    #[test]
    fn test_bose_einstein_high_temperature() {
        // High-T limit: n ≈ k_B T / (E - μ) >> 1
        let mu = 0.0;
        let e = K_B * 1.0; // very small energy gap
        let n = StatMech::bose_einstein(e, mu, 1e6);
        assert!(n.is_some());
        assert!(n.expect("should succeed") > 1.0);
    }

    #[test]
    fn test_bose_einstein_mu_equals_e_returns_none() {
        let e = 1e-19;
        let result = StatMech::bose_einstein(e, e, 300.0);
        assert!(result.is_none());
    }

    #[test]
    fn test_bose_einstein_zero_temperature_returns_none() {
        let result = StatMech::bose_einstein(1e-19, 0.0, 0.0);
        assert!(result.is_none());
    }

    #[test]
    fn test_bose_einstein_mu_above_e_returns_none() {
        // μ > E is unphysical for bosons → exp becomes negative exponent < 1 → denom < 0
        let result = StatMech::bose_einstein(1e-20, 1e-19, 300.0);
        assert!(result.is_none());
    }

    #[test]
    fn test_bose_einstein_occupation_positive() {
        let e = 1e-19;
        let mu = -1e-20; // μ < E (physical)
        let n = StatMech::bose_einstein(e, mu, 300.0);
        assert!(n.is_some());
        assert!(n.expect("should succeed") > 0.0);
    }

    // ── Mean free path ────────────────────────────────────────────────────────

    #[test]
    fn test_mean_free_path_nitrogen_stp() {
        // N₂ at STP: d ≈ 370 pm, n ≈ 2.69e25 m⁻³
        let d = 370e-12_f64;
        let n = 2.69e25_f64;
        let lambda = StatMech::mean_free_path(d, n).expect("should succeed");
        // Expected ≈ 60–70 nm
        assert!(lambda > 50e-9 && lambda < 100e-9);
    }

    #[test]
    fn test_mean_free_path_from_pressure() {
        // N₂ at STP: T=273K, P=101325 Pa, d=370 pm
        let lambda = StatMech::mean_free_path_from_pressure(370e-12, 101_325.0, 273.15)
            .expect("should succeed");
        assert!(lambda > 30e-9 && lambda < 120e-9);
    }

    #[test]
    fn test_mean_free_path_zero_diameter_none() {
        assert!(StatMech::mean_free_path(0.0, 1e25).is_none());
    }

    #[test]
    fn test_mean_free_path_zero_density_none() {
        assert!(StatMech::mean_free_path(370e-12, 0.0).is_none());
    }

    #[test]
    fn test_mean_free_path_negative_inputs_none() {
        assert!(StatMech::mean_free_path(-1e-10, 1e25).is_none());
    }

    #[test]
    fn test_mean_free_path_increases_with_lower_density() {
        let d = 370e-12_f64;
        let lambda_high = StatMech::mean_free_path(d, 1e25).expect("should succeed");
        let lambda_low = StatMech::mean_free_path(d, 1e20).expect("should succeed");
        assert!(lambda_low > lambda_high);
    }

    // ── Maxwell-Boltzmann distribution ────────────────────────────────────────

    #[test]
    fn test_mb_most_probable_speed_n2_300k() {
        let dist = MaxwellBoltzmannDist::new(MASS_N2, 300.0);
        let v_p = dist.most_probable_speed();
        // Expected: √(2RT/M) ≈ 422 m/s for N₂
        assert!((v_p - 422.0).abs() < 10.0);
    }

    #[test]
    fn test_mb_mean_speed_n2_300k() {
        let dist = MaxwellBoltzmannDist::new(MASS_N2, 300.0);
        let v_mean = dist.mean_speed();
        // Expected: √(8RT/(πM)) ≈ 476 m/s for N₂
        assert!((v_mean - 476.0).abs() < 15.0);
    }

    #[test]
    fn test_mb_rms_speed_n2_300k() {
        let dist = MaxwellBoltzmannDist::new(MASS_N2, 300.0);
        let v_rms = dist.rms_speed();
        // Expected: √(3RT/M) ≈ 517 m/s for N₂
        assert!((v_rms - 517.0).abs() < 15.0);
    }

    #[test]
    fn test_mb_speed_hierarchy() {
        let dist = MaxwellBoltzmannDist::new(MASS_N2, 300.0);
        // v_p < v_mean < v_rms always
        assert!(dist.most_probable_speed() < dist.mean_speed());
        assert!(dist.mean_speed() < dist.rms_speed());
    }

    #[test]
    fn test_mb_pdf_zero_at_zero_speed() {
        let dist = MaxwellBoltzmannDist::new(MASS_N2, 300.0);
        assert!((dist.pdf(0.0)).abs() < 1e-10);
    }

    #[test]
    fn test_mb_pdf_positive_at_typical_speed() {
        let dist = MaxwellBoltzmannDist::new(MASS_N2, 300.0);
        assert!(dist.pdf(400.0) > 0.0);
    }

    #[test]
    fn test_mb_pdf_negative_speed_zero() {
        let dist = MaxwellBoltzmannDist::new(MASS_N2, 300.0);
        assert_eq!(dist.pdf(-1.0), 0.0);
    }

    #[test]
    fn test_mb_cdf_zero_at_zero() {
        let dist = MaxwellBoltzmannDist::new(MASS_N2, 300.0);
        assert!((dist.cdf(0.0)).abs() < 1e-10);
    }

    #[test]
    fn test_mb_cdf_approaches_one_at_high_speed() {
        let dist = MaxwellBoltzmannDist::new(MASS_N2, 300.0);
        let cdf = dist.cdf(5000.0); // 5000 m/s >> v_rms
        assert!(cdf > 0.99);
    }

    #[test]
    fn test_mb_cdf_monotone() {
        let dist = MaxwellBoltzmannDist::new(MASS_N2, 300.0);
        let v1 = 200.0;
        let v2 = 500.0;
        assert!(dist.cdf(v2) > dist.cdf(v1));
    }

    #[test]
    fn test_mb_thermal_speed() {
        let dist = MaxwellBoltzmannDist::new(MASS_N2, 300.0);
        let a = dist.thermal_speed();
        let expected = (K_B * 300.0 / MASS_N2).sqrt();
        assert!((a - expected).abs() < 1e-3);
    }

    // ── erf helper ────────────────────────────────────────────────────────────

    #[test]
    fn test_erf_at_zero() {
        assert!((erf(0.0)).abs() < 1e-7);
    }

    #[test]
    fn test_erf_at_large_x_approaches_one() {
        assert!((erf(5.0) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_erf_symmetry() {
        let x = 1.2;
        assert!((erf(-x) + erf(x)).abs() < 1e-7);
    }

    // ── Physical constant checks ──────────────────────────────────────────────

    #[test]
    fn test_boltzmann_constant_value() {
        assert!((K_B - 1.380_649e-23).abs() < 1e-30);
    }

    #[test]
    fn test_avogadro_constant_value() {
        assert!((StatMech::N_A - 6.022_140_76e23).abs() < 1e16);
    }
}
