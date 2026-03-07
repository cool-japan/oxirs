//! # Electromagnetics Module
//!
//! Electromagnetic field computations including Coulomb's law, superposition,
//! Biot-Savart solenoid fields, Lorentz force, skin depth, and circuit transients.
//!
//! ## Physical Constants
//!
//! - Coulomb constant: `K_E = 8.987551787e9 N·m²/C²`
//! - Vacuum permeability: `MU_0 = 1.2566370614e-6 H/m`
//! - Vacuum permittivity: `EPSILON_0 = 8.854187817e-12 F/m`

use std::f64::consts::PI;

/// Electric field vector in Cartesian coordinates (V/m)
#[derive(Debug, Clone, PartialEq)]
pub struct ElectricField {
    pub ex: f64,
    pub ey: f64,
    pub ez: f64,
}

/// Magnetic field vector in Cartesian coordinates (T = Tesla)
#[derive(Debug, Clone, PartialEq)]
pub struct MagneticField {
    pub bx: f64,
    pub by: f64,
    pub bz: f64,
}

/// Point charge in 3D space
#[derive(Debug, Clone)]
pub struct PointCharge {
    /// Charge in Coulombs
    pub charge_c: f64,
    /// X position in meters
    pub x: f64,
    /// Y position in meters
    pub y: f64,
    /// Z position in meters
    pub z: f64,
}

/// Current-carrying loop (for magnetic field computation)
#[derive(Debug, Clone)]
pub struct CurrentLoop {
    /// Current in Amperes
    pub current_a: f64,
    /// Loop radius in meters
    pub radius_m: f64,
    /// Center position (x, y, z)
    pub center: (f64, f64, f64),
    /// Axis direction unit vector (x, y, z)
    pub axis: (f64, f64, f64),
}

/// Main calculator for electromagnetic field computations
pub struct ElectromagneticsCalculator;

impl ElectromagneticsCalculator {
    /// Coulomb constant: K_E = 1 / (4π ε₀) ≈ 8.987551787 × 10⁹ N·m²/C²
    pub const K_E: f64 = 8.987_551_787e9;

    /// Vacuum permeability: μ₀ ≈ 1.2566370614 × 10⁻⁶ H/m
    pub const MU_0: f64 = 1.256_637_061_4e-6;

    /// Vacuum permittivity: ε₀ = 1 / (μ₀ c²) ≈ 8.854187817 × 10⁻¹² F/m
    pub const EPSILON_0: f64 = 8.854_187_817e-12;

    /// Speed of light: c ≈ 2.997924458 × 10⁸ m/s
    pub const C: f64 = 2.997_924_458e8;

    /// Compute the electric field at point (x, y, z) due to a single point charge.
    ///
    /// Uses Coulomb's law: **E** = K_E · q / r² · r̂
    ///
    /// Returns a zero field if the evaluation point coincides with the charge position
    /// (to avoid division by zero).
    pub fn electric_field(charge: &PointCharge, x: f64, y: f64, z: f64) -> ElectricField {
        let dx = x - charge.x;
        let dy = y - charge.y;
        let dz = z - charge.z;
        let r_sq = dx * dx + dy * dy + dz * dz;

        if r_sq < f64::EPSILON {
            return ElectricField::zero();
        }

        let r = r_sq.sqrt();
        let factor = Self::K_E * charge.charge_c / (r_sq * r);

        ElectricField {
            ex: factor * dx,
            ey: factor * dy,
            ez: factor * dz,
        }
    }

    /// Scalar Coulomb force magnitude between two point charges separated by distance `r`.
    ///
    /// F = K_E · |q1| · |q2| / r²
    ///
    /// Returns 0.0 if `r <= 0`.
    pub fn coulomb_force(q1: f64, q2: f64, r: f64) -> f64 {
        if r <= 0.0 {
            return 0.0;
        }
        Self::K_E * q1.abs() * q2.abs() / (r * r)
    }

    /// Superposition of electric fields from multiple point charges at (x, y, z).
    ///
    /// E_total = Σ E_i(x, y, z)
    pub fn superposition(charges: &[PointCharge], x: f64, y: f64, z: f64) -> ElectricField {
        let mut total = ElectricField::zero();
        for charge in charges {
            let e = Self::electric_field(charge, x, y, z);
            total = total.add(&e);
        }
        total
    }

    /// Magnetic field **on the axis** of an ideal solenoid at its center.
    ///
    /// Uses the solenoid formula: B = μ₀ · n · I = μ₀ · (N/L) · I
    ///
    /// - `n_turns`: total number of turns N
    /// - `current_a`: current in Amperes
    /// - `length_m`: solenoid length in meters
    ///
    /// Returns B in Tesla (T).
    pub fn solenoid_field_axial(n_turns: f64, current_a: f64, length_m: f64) -> f64 {
        if length_m <= 0.0 {
            return 0.0;
        }
        let n = n_turns / length_m; // turns per meter
        Self::MU_0 * n * current_a
    }

    /// Lorentz force on a charge moving through combined electric and magnetic fields.
    ///
    /// **F** = q(**E** + **v** × **B**)
    ///
    /// Returns force vector (Fx, Fy, Fz) in Newtons.
    pub fn lorentz_force(
        q: f64,
        e: &ElectricField,
        b: &MagneticField,
        v: (f64, f64, f64),
    ) -> (f64, f64, f64) {
        let (vx, vy, vz) = v;

        // Cross product: v × B
        let cross_x = vy * b.bz - vz * b.by;
        let cross_y = vz * b.bx - vx * b.bz;
        let cross_z = vx * b.by - vy * b.bx;

        let fx = q * (e.ex + cross_x);
        let fy = q * (e.ey + cross_y);
        let fz = q * (e.ez + cross_z);

        (fx, fy, fz)
    }

    /// Energy stored in an electric field over a volume.
    ///
    /// U = ½ · ε₀ · E² · V
    ///
    /// - `e_magnitude`: electric field magnitude in V/m
    /// - `volume_m3`: volume in m³
    ///
    /// Returns energy in Joules.
    pub fn electric_field_energy(e_magnitude: f64, volume_m3: f64) -> f64 {
        0.5 * Self::EPSILON_0 * e_magnitude * e_magnitude * volume_m3
    }

    /// Capacitor voltage during discharge through a resistor.
    ///
    /// V(t) = V₀ · exp(−t / RC)
    ///
    /// - `v0`: initial voltage in Volts
    /// - `c`: capacitance in Farads
    /// - `r`: resistance in Ohms
    /// - `t`: time in seconds
    pub fn capacitor_voltage(v0: f64, c: f64, r: f64, t: f64) -> f64 {
        if r <= 0.0 || c <= 0.0 {
            return v0;
        }
        let tau = r * c;
        v0 * (-t / tau).exp()
    }

    /// Inductor current rise in an RL circuit driven by constant voltage V.
    ///
    /// I(t) = (V / R) · (1 − exp(−R·t / L))
    ///
    /// - `v`: supply voltage in Volts
    /// - `l`: inductance in Henrys
    /// - `r`: resistance in Ohms
    /// - `t`: time in seconds
    pub fn inductor_current(v: f64, l: f64, r: f64, t: f64) -> f64 {
        if r <= 0.0 || l <= 0.0 {
            return 0.0;
        }
        (v / r) * (1.0 - (-r * t / l).exp())
    }

    /// Skin depth in a conductor.
    ///
    /// δ = √(2ρ / (ω · μ₀))
    ///
    /// where ω = 2π · f
    ///
    /// - `resistivity`: electrical resistivity in Ω·m
    /// - `frequency_hz`: frequency in Hz
    ///
    /// Returns skin depth in meters.
    pub fn skin_depth(resistivity: f64, frequency_hz: f64) -> f64 {
        if frequency_hz <= 0.0 || resistivity < 0.0 {
            return f64::INFINITY;
        }
        let omega = 2.0 * PI * frequency_hz;
        (2.0 * resistivity / (omega * Self::MU_0)).sqrt()
    }

    /// Magnetic field on axis of a single current loop at axial distance z from center.
    ///
    /// Uses Biot-Savart for a circular loop:
    /// B_z(z) = μ₀ · I · R² / (2 · (R² + z²)^(3/2))
    ///
    /// - `loop_`: the current loop definition
    /// - `axial_distance`: distance along the axis from loop center (m)
    pub fn loop_field_axial(loop_: &CurrentLoop, axial_distance: f64) -> f64 {
        let r = loop_.radius_m;
        let i = loop_.current_a;
        let r_sq = r * r;
        let z_sq = axial_distance * axial_distance;
        let denom = (r_sq + z_sq).powf(1.5);
        if denom < f64::EPSILON {
            return 0.0;
        }
        Self::MU_0 * i * r_sq / (2.0 * denom)
    }
}

impl ElectricField {
    /// Magnitude of the electric field vector |**E**| = √(Ex² + Ey² + Ez²)
    pub fn magnitude(&self) -> f64 {
        (self.ex * self.ex + self.ey * self.ey + self.ez * self.ez).sqrt()
    }

    /// Zero electric field (no field)
    pub fn zero() -> Self {
        ElectricField {
            ex: 0.0,
            ey: 0.0,
            ez: 0.0,
        }
    }

    /// Vector addition of two electric fields (superposition principle)
    pub fn add(&self, other: &ElectricField) -> ElectricField {
        ElectricField {
            ex: self.ex + other.ex,
            ey: self.ey + other.ey,
            ez: self.ez + other.ez,
        }
    }
}

impl MagneticField {
    /// Magnitude of the magnetic field vector |**B**| = √(Bx² + By² + Bz²)
    pub fn magnitude(&self) -> f64 {
        (self.bx * self.bx + self.by * self.by + self.bz * self.bz).sqrt()
    }

    /// Zero magnetic field
    pub fn zero() -> Self {
        MagneticField {
            bx: 0.0,
            by: 0.0,
            bz: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-6;

    // ===== ElectricField helper tests =====

    #[test]
    fn test_electric_field_magnitude_zero() {
        let e = ElectricField::zero();
        assert_eq!(e.magnitude(), 0.0);
    }

    #[test]
    fn test_electric_field_magnitude_unit() {
        let e = ElectricField { ex: 1.0, ey: 0.0, ez: 0.0 };
        assert!((e.magnitude() - 1.0).abs() < TOL);
    }

    #[test]
    fn test_electric_field_magnitude_3d() {
        let e = ElectricField { ex: 3.0, ey: 4.0, ez: 0.0 };
        assert!((e.magnitude() - 5.0).abs() < TOL);
    }

    #[test]
    fn test_electric_field_add() {
        let e1 = ElectricField { ex: 1.0, ey: 2.0, ez: 3.0 };
        let e2 = ElectricField { ex: 4.0, ey: 5.0, ez: 6.0 };
        let sum = e1.add(&e2);
        assert!((sum.ex - 5.0).abs() < TOL);
        assert!((sum.ey - 7.0).abs() < TOL);
        assert!((sum.ez - 9.0).abs() < TOL);
    }

    #[test]
    fn test_electric_field_add_zero() {
        let e = ElectricField { ex: 3.0, ey: -1.0, ez: 2.5 };
        let zero = ElectricField::zero();
        let sum = e.add(&zero);
        assert!((sum.ex - e.ex).abs() < TOL);
        assert!((sum.ey - e.ey).abs() < TOL);
        assert!((sum.ez - e.ez).abs() < TOL);
    }

    // ===== MagneticField helper tests =====

    #[test]
    fn test_magnetic_field_magnitude_zero() {
        let b = MagneticField::zero();
        assert_eq!(b.magnitude(), 0.0);
    }

    #[test]
    fn test_magnetic_field_magnitude_345() {
        let b = MagneticField { bx: 3.0, by: 4.0, bz: 0.0 };
        assert!((b.magnitude() - 5.0).abs() < TOL);
    }

    // ===== Coulomb's law / electric_field =====

    #[test]
    fn test_electric_field_single_charge_at_distance_1() {
        // Charge q=1C at origin; evaluate at (1, 0, 0)
        let charge = PointCharge { charge_c: 1.0, x: 0.0, y: 0.0, z: 0.0 };
        let e = ElectromagneticsCalculator::electric_field(&charge, 1.0, 0.0, 0.0);
        // |E| should be K_E * 1 / 1^2 = K_E
        let expected = ElectromagneticsCalculator::K_E;
        assert!((e.magnitude() - expected).abs() / expected < 1e-9);
        assert!((e.ex - expected).abs() / expected < 1e-9);
        assert!(e.ey.abs() < TOL);
        assert!(e.ez.abs() < TOL);
    }

    #[test]
    fn test_electric_field_inverse_square_law() {
        // |E| ∝ 1/r² — doubling r should quarter the field magnitude
        let charge = PointCharge { charge_c: 1.0e-6, x: 0.0, y: 0.0, z: 0.0 };
        let e1 = ElectromagneticsCalculator::electric_field(&charge, 1.0, 0.0, 0.0);
        let e2 = ElectromagneticsCalculator::electric_field(&charge, 2.0, 0.0, 0.0);
        let ratio = e1.magnitude() / e2.magnitude();
        assert!((ratio - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_electric_field_negative_charge_direction() {
        // Negative charge: field should point toward the charge (negative x direction)
        let charge = PointCharge { charge_c: -1.0, x: 0.0, y: 0.0, z: 0.0 };
        let e = ElectromagneticsCalculator::electric_field(&charge, 1.0, 0.0, 0.0);
        assert!(e.ex < 0.0, "field should point toward negative charge");
    }

    #[test]
    fn test_electric_field_at_charge_location_is_zero() {
        let charge = PointCharge { charge_c: 5.0, x: 3.0, y: 3.0, z: 3.0 };
        let e = ElectromagneticsCalculator::electric_field(&charge, 3.0, 3.0, 3.0);
        assert_eq!(e.magnitude(), 0.0);
    }

    #[test]
    fn test_electric_field_symmetry() {
        // Charge at origin: field at (2,0,0) and (-2,0,0) should have opposite Ex, same |E|
        let charge = PointCharge { charge_c: 1.0, x: 0.0, y: 0.0, z: 0.0 };
        let e_pos = ElectromagneticsCalculator::electric_field(&charge, 2.0, 0.0, 0.0);
        let e_neg = ElectromagneticsCalculator::electric_field(&charge, -2.0, 0.0, 0.0);
        assert!((e_pos.magnitude() - e_neg.magnitude()).abs() < TOL);
        assert!((e_pos.ex + e_neg.ex).abs() < TOL);
    }

    // ===== Coulomb force scalar =====

    #[test]
    fn test_coulomb_force_unit_charges() {
        // Two 1 C charges 1 m apart: F = K_E
        let f = ElectromagneticsCalculator::coulomb_force(1.0, 1.0, 1.0);
        assert!((f - ElectromagneticsCalculator::K_E).abs() / ElectromagneticsCalculator::K_E < 1e-9);
    }

    #[test]
    fn test_coulomb_force_inverse_square() {
        // Double the distance → quarter the force
        let f1 = ElectromagneticsCalculator::coulomb_force(1.0, 1.0, 1.0);
        let f2 = ElectromagneticsCalculator::coulomb_force(1.0, 1.0, 2.0);
        let ratio = f1 / f2;
        assert!((ratio - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_coulomb_force_zero_distance_returns_zero() {
        let f = ElectromagneticsCalculator::coulomb_force(1.0, 1.0, 0.0);
        assert_eq!(f, 0.0);
    }

    #[test]
    fn test_coulomb_force_always_positive() {
        // Force magnitude is always positive regardless of sign conventions
        let f = ElectromagneticsCalculator::coulomb_force(-2.0, -3.0, 0.5);
        assert!(f > 0.0);
    }

    // ===== Superposition =====

    #[test]
    fn test_superposition_two_opposite_charges_cancel_on_axis() {
        // +q at (-d, 0, 0) and -q at (+d, 0, 0) cancel at origin
        let charges = vec![
            PointCharge { charge_c: 1.0, x: -1.0, y: 0.0, z: 0.0 },
            PointCharge { charge_c: -1.0, x: 1.0, y: 0.0, z: 0.0 },
        ];
        let e = ElectromagneticsCalculator::superposition(&charges, 0.0, 0.0, 0.0);
        // Ex should be non-zero (pointing toward −q), Ey, Ez zero by symmetry
        assert!(e.ey.abs() < TOL);
        assert!(e.ez.abs() < TOL);
    }

    #[test]
    fn test_superposition_two_equal_charges_double_field() {
        // Two identical charges at same position → 2× single charge field
        let single = PointCharge { charge_c: 1.0e-9, x: 0.0, y: 0.0, z: 0.0 };
        let double = vec![
            PointCharge { charge_c: 1.0e-9, x: 0.0, y: 0.0, z: 0.0 },
            PointCharge { charge_c: 1.0e-9, x: 0.0, y: 0.0, z: 0.0 },
        ];
        let e1 = ElectromagneticsCalculator::electric_field(&single, 1.0, 0.0, 0.0);
        let e2 = ElectromagneticsCalculator::superposition(&double, 1.0, 0.0, 0.0);
        assert!((e2.magnitude() - 2.0 * e1.magnitude()).abs() / e1.magnitude() < 1e-9);
    }

    #[test]
    fn test_superposition_empty_charges() {
        let charges: Vec<PointCharge> = vec![];
        let e = ElectromagneticsCalculator::superposition(&charges, 1.0, 2.0, 3.0);
        assert_eq!(e.magnitude(), 0.0);
    }

    #[test]
    fn test_superposition_three_charges() {
        let charges = vec![
            PointCharge { charge_c: 1.0e-9, x: 0.0, y: 0.0, z: 0.0 },
            PointCharge { charge_c: 2.0e-9, x: 1.0, y: 0.0, z: 0.0 },
            PointCharge { charge_c: -1.0e-9, x: -1.0, y: 0.0, z: 0.0 },
        ];
        let e = ElectromagneticsCalculator::superposition(&charges, 5.0, 0.0, 0.0);
        // Just verify it runs and field is finite
        assert!(e.magnitude().is_finite());
    }

    // ===== Electric field energy =====

    #[test]
    fn test_electric_field_energy_zero_volume() {
        let u = ElectromagneticsCalculator::electric_field_energy(1000.0, 0.0);
        assert_eq!(u, 0.0);
    }

    #[test]
    fn test_electric_field_energy_proportional_to_e_squared() {
        let u1 = ElectromagneticsCalculator::electric_field_energy(100.0, 1.0);
        let u2 = ElectromagneticsCalculator::electric_field_energy(200.0, 1.0);
        // u2 should be 4× u1
        assert!((u2 / u1 - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_electric_field_energy_proportional_to_volume() {
        let u1 = ElectromagneticsCalculator::electric_field_energy(100.0, 1.0);
        let u2 = ElectromagneticsCalculator::electric_field_energy(100.0, 3.0);
        assert!((u2 / u1 - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_electric_field_energy_positive() {
        let u = ElectromagneticsCalculator::electric_field_energy(1.0e3, 0.001);
        assert!(u > 0.0);
    }

    // ===== Capacitor discharge =====

    #[test]
    fn test_capacitor_voltage_at_t0() {
        // V(0) = V0 * exp(0) = V0
        let v = ElectromagneticsCalculator::capacitor_voltage(12.0, 1.0e-3, 1000.0, 0.0);
        assert!((v - 12.0).abs() < TOL);
    }

    #[test]
    fn test_capacitor_voltage_at_one_tau() {
        // V(RC) = V0 * exp(-1) ≈ 0.3679 * V0
        let v0 = 10.0;
        let r = 1000.0;
        let c = 1.0e-3;
        let tau = r * c;
        let v = ElectromagneticsCalculator::capacitor_voltage(v0, c, r, tau);
        let expected = v0 * (-1.0f64).exp();
        assert!((v - expected).abs() / expected < 1e-9);
    }

    #[test]
    fn test_capacitor_voltage_decays_monotonically() {
        let v0 = 5.0;
        let r = 100.0;
        let c = 1.0e-4;
        let v1 = ElectromagneticsCalculator::capacitor_voltage(v0, c, r, 0.001);
        let v2 = ElectromagneticsCalculator::capacitor_voltage(v0, c, r, 0.002);
        assert!(v2 < v1);
    }

    #[test]
    fn test_capacitor_voltage_approaches_zero() {
        let v = ElectromagneticsCalculator::capacitor_voltage(100.0, 1.0e-3, 1.0, 1000.0);
        assert!(v < 1.0e-10);
    }

    #[test]
    fn test_capacitor_voltage_37_percent_at_tau() {
        // At t = RC, voltage ≈ 37% of V0
        let v0 = 100.0;
        let r = 1.0;
        let c = 1.0;
        let v = ElectromagneticsCalculator::capacitor_voltage(v0, c, r, r * c);
        let pct = v / v0;
        assert!((pct - 0.367879441).abs() < 1e-7);
    }

    // ===== Inductor current rise =====

    #[test]
    fn test_inductor_current_at_t0() {
        // I(0) = 0
        let i = ElectromagneticsCalculator::inductor_current(12.0, 0.01, 100.0, 0.0);
        assert!(i.abs() < TOL);
    }

    #[test]
    fn test_inductor_current_at_one_tau() {
        // I(L/R) = V/R * (1 - 1/e) ≈ 0.6321 * V/R
        let v = 10.0;
        let l = 0.1;
        let r = 10.0;
        let tau = l / r;
        let i = ElectromagneticsCalculator::inductor_current(v, l, r, tau);
        let expected = (v / r) * (1.0 - (-1.0f64).exp());
        assert!((i - expected).abs() / expected < 1e-9);
    }

    #[test]
    fn test_inductor_current_approaches_steady_state() {
        // I(∞) → V/R
        let v = 5.0;
        let l = 1.0;
        let r = 1.0;
        let i = ElectromagneticsCalculator::inductor_current(v, l, r, 1000.0);
        assert!((i - v / r).abs() < 1.0e-9);
    }

    #[test]
    fn test_inductor_current_monotone_rise() {
        let v = 12.0;
        let l = 0.05;
        let r = 50.0;
        let i1 = ElectromagneticsCalculator::inductor_current(v, l, r, 0.0005);
        let i2 = ElectromagneticsCalculator::inductor_current(v, l, r, 0.001);
        assert!(i2 > i1);
    }

    #[test]
    fn test_inductor_current_63_percent_at_tau() {
        // At t = L/R, current ≈ 63.21% of V/R
        let v = 100.0;
        let l = 1.0;
        let r = 1.0;
        let tau = l / r;
        let i = ElectromagneticsCalculator::inductor_current(v, l, r, tau);
        let pct = i / (v / r);
        assert!((pct - 0.632120559).abs() < 1e-7);
    }

    // ===== Skin depth =====

    #[test]
    fn test_skin_depth_copper_at_50hz() {
        // Copper: ρ ≈ 1.7e-8 Ω·m, f=50 Hz
        // δ = sqrt(2 * 1.7e-8 / (2π * 50 * 1.2566e-6)) ≈ 9.27 mm
        let delta = ElectromagneticsCalculator::skin_depth(1.7e-8, 50.0);
        assert!(delta > 0.005 && delta < 0.02, "copper 50Hz skin depth should be ~9mm, got {:.4} m", delta);
    }

    #[test]
    fn test_skin_depth_decreases_with_frequency() {
        let d1 = ElectromagneticsCalculator::skin_depth(1.7e-8, 50.0);
        let d2 = ElectromagneticsCalculator::skin_depth(1.7e-8, 50000.0);
        assert!(d2 < d1, "higher frequency → smaller skin depth");
    }

    #[test]
    fn test_skin_depth_increases_with_resistivity() {
        let d1 = ElectromagneticsCalculator::skin_depth(1.7e-8, 1000.0); // copper
        let d2 = ElectromagneticsCalculator::skin_depth(1.0e-6, 1000.0); // poor conductor
        assert!(d2 > d1);
    }

    #[test]
    fn test_skin_depth_formula_ratio() {
        // Quadrupling frequency → halving skin depth (δ ∝ 1/√f)
        let d1 = ElectromagneticsCalculator::skin_depth(1.0e-7, 100.0);
        let d2 = ElectromagneticsCalculator::skin_depth(1.0e-7, 400.0);
        let ratio = d1 / d2;
        assert!((ratio - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_skin_depth_zero_frequency_returns_infinity() {
        let d = ElectromagneticsCalculator::skin_depth(1.0e-7, 0.0);
        assert!(d.is_infinite());
    }

    // ===== Solenoid field =====

    #[test]
    fn test_solenoid_field_proportional_to_n_times_i_over_l() {
        // B = μ₀ * n * I where n = N/L
        // Doubling N (at same L and I) should double B
        let b1 = ElectromagneticsCalculator::solenoid_field_axial(100.0, 1.0, 0.1);
        let b2 = ElectromagneticsCalculator::solenoid_field_axial(200.0, 1.0, 0.1);
        assert!((b2 / b1 - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_solenoid_field_proportional_to_current() {
        let b1 = ElectromagneticsCalculator::solenoid_field_axial(1000.0, 1.0, 1.0);
        let b2 = ElectromagneticsCalculator::solenoid_field_axial(1000.0, 3.0, 1.0);
        assert!((b2 / b1 - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_solenoid_field_1000_turns_1m_1a() {
        // n=1000 turns/m, I=1 A → B = μ₀ * 1000 ≈ 1.2566e-3 T
        let b = ElectromagneticsCalculator::solenoid_field_axial(1000.0, 1.0, 1.0);
        let expected = ElectromagneticsCalculator::MU_0 * 1000.0;
        assert!((b - expected).abs() / expected < 1e-9);
    }

    #[test]
    fn test_solenoid_field_zero_length_returns_zero() {
        let b = ElectromagneticsCalculator::solenoid_field_axial(1000.0, 1.0, 0.0);
        assert_eq!(b, 0.0);
    }

    // ===== Lorentz force =====

    #[test]
    fn test_lorentz_force_electric_only() {
        // v=0, E=(1,0,0), B=(0,0,0): F = q * E
        let e = ElectricField { ex: 1000.0, ey: 0.0, ez: 0.0 };
        let b = MagneticField::zero();
        let (fx, fy, fz) = ElectromagneticsCalculator::lorentz_force(1.0e-6, &e, &b, (0.0, 0.0, 0.0));
        assert!((fx - 1.0e-3).abs() < 1e-12);
        assert!(fy.abs() < TOL);
        assert!(fz.abs() < TOL);
    }

    #[test]
    fn test_lorentz_force_magnetic_cross_product() {
        // v=(1,0,0), B=(0,0,1): v×B = (0,-1,0) → F = q(0,-B,0)
        let e = ElectricField::zero();
        let b = MagneticField { bx: 0.0, by: 0.0, bz: 1.0 };
        let q = 2.0;
        let (fx, fy, fz) = ElectromagneticsCalculator::lorentz_force(q, &e, &b, (1.0, 0.0, 0.0));
        assert!(fx.abs() < TOL);
        assert!((fy - (-2.0)).abs() < TOL); // q*(vy*Bz - vz*By) = q*(0-0)=0, q*(vz*Bx - vx*Bz)=q*(-1)
        assert!(fz.abs() < TOL);
    }

    #[test]
    fn test_lorentz_force_both_fields() {
        let e = ElectricField { ex: 100.0, ey: 0.0, ez: 0.0 };
        let b = MagneticField { bx: 0.0, by: 0.0, bz: 1.0 };
        let q = 1.0e-6;
        let v = (1.0e4, 0.0, 0.0);
        let (fx, fy, fz) = ElectromagneticsCalculator::lorentz_force(q, &e, &b, v);
        // fx = q*(Ex + vy*Bz - vz*By) = q*(100 + 0 - 0)
        assert!((fx - q * 100.0).abs() < 1e-18);
        // fy = q*(Ey + vz*Bx - vx*Bz) = q*(0 + 0 - 1e4*1) = -q*1e4
        assert!((fy - q * (-1.0e4)).abs() < 1e-18);
        assert!(fz.abs() < 1e-18);
    }

    #[test]
    fn test_lorentz_force_zero_charge() {
        let e = ElectricField { ex: 1.0e6, ey: 1.0e6, ez: 1.0e6 };
        let b = MagneticField { bx: 1.0, by: 1.0, bz: 1.0 };
        let (fx, fy, fz) = ElectromagneticsCalculator::lorentz_force(0.0, &e, &b, (1000.0, 1000.0, 1000.0));
        assert_eq!(fx, 0.0);
        assert_eq!(fy, 0.0);
        assert_eq!(fz, 0.0);
    }

    // ===== Current loop field =====

    #[test]
    fn test_loop_field_at_center() {
        // B_z(0) = μ₀*I / (2*R)
        let loop_ = CurrentLoop {
            current_a: 1.0,
            radius_m: 0.1,
            center: (0.0, 0.0, 0.0),
            axis: (0.0, 0.0, 1.0),
        };
        let b = ElectromagneticsCalculator::loop_field_axial(&loop_, 0.0);
        let expected = ElectromagneticsCalculator::MU_0 * 1.0 / (2.0 * 0.1);
        assert!((b - expected).abs() / expected < 1e-9);
    }

    #[test]
    fn test_loop_field_decreases_with_distance() {
        let loop_ = CurrentLoop {
            current_a: 10.0,
            radius_m: 0.05,
            center: (0.0, 0.0, 0.0),
            axis: (0.0, 0.0, 1.0),
        };
        let b0 = ElectromagneticsCalculator::loop_field_axial(&loop_, 0.0);
        let b1 = ElectromagneticsCalculator::loop_field_axial(&loop_, 0.1);
        assert!(b1 < b0);
    }

    #[test]
    fn test_loop_field_proportional_to_current() {
        let make_loop = |i: f64| CurrentLoop {
            current_a: i,
            radius_m: 0.05,
            center: (0.0, 0.0, 0.0),
            axis: (0.0, 0.0, 1.0),
        };
        let b1 = ElectromagneticsCalculator::loop_field_axial(&make_loop(1.0), 0.0);
        let b2 = ElectromagneticsCalculator::loop_field_axial(&make_loop(5.0), 0.0);
        assert!((b2 / b1 - 5.0).abs() < 1e-9);
    }
}
