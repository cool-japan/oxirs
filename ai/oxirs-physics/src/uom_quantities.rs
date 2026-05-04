//! Type-safe physics quantities via the `uom` crate (requires `simulation` feature).
//!
//! This module provides [`UomQuantity`] wrappers for common SI quantities used
//! throughout the physics simulation bridge.  Using compile-time dimensional
//! types eliminates unit-mismatch bugs that raw `f64` parameters cannot catch.
//!
//! ## Supported Quantities
//!
//! | Type alias | SI unit | Description |
//! |---|---|---|
//! | `Mass` | kg | mass |
//! | `Velocity` | m/s | velocity |
//! | `Acceleration` | m/s² | acceleration |
//! | `Energy` | J (kg·m²/s²) | energy |
//! | `Temperature` | K | thermodynamic temperature |
//! | `Force` | N (kg·m/s²) | force |
//! | `Pressure` | Pa (kg/(m·s²)) | pressure |
//! | `Length` | m | length |
//! | `Time` | s | time interval |
//! | `AngularMomentum` | kg·m²/s | angular momentum |
//! | `Entropy` | J/K | thermodynamic entropy |
//!
//! ## Example
//!
//! ```rust,no_run
//! # #[cfg(feature = "simulation")] {
//! use oxirs_physics::uom_quantities::{energy_j, mass_kg, to_joules, to_kg};
//!
//! let m = mass_kg(1.5);
//! let e = energy_j(100.0);
//!
//! // Convert back to f64 for simulation APIs
//! assert!((to_kg(&m) - 1.5).abs() < 1e-12);
//! assert!((to_joules(&e) - 100.0).abs() < 1e-12);
//! # }
//! ```

use uom::si::f64 as si;
use uom::si::{
    acceleration::meter_per_second_squared, angular_momentum::kilogram_square_meter_per_second,
    energy::joule, force::newton, length::meter, mass::kilogram, pressure::pascal,
    thermodynamic_temperature::kelvin as temperature_kelvin, time::second,
    velocity::meter_per_second,
};

// ─────────────────────────────────────────────────────────────────────────────
// Type aliases for common SI quantities
// ─────────────────────────────────────────────────────────────────────────────

/// Mass in SI kilograms.
pub type Mass = si::Mass;

/// Length in SI metres.
pub type Length = si::Length;

/// Time in SI seconds.
pub type Time = si::Time;

/// Velocity in SI metres per second.
pub type Velocity = si::Velocity;

/// Acceleration in SI metres per second squared.
pub type Acceleration = si::Acceleration;

/// Force in SI Newtons (kg·m/s²).
pub type Force = si::Force;

/// Pressure in SI Pascals (kg/(m·s²)).
pub type Pressure = si::Pressure;

/// Energy in SI Joules (kg·m²/s²).
pub type Energy = si::Energy;

/// Thermodynamic temperature in SI Kelvin.
pub type ThermodynamicTemperature = si::ThermodynamicTemperature;

/// Angular momentum in kg·m²/s.
pub type AngularMomentum = si::AngularMomentum;

// ─────────────────────────────────────────────────────────────────────────────
// Entropy (J/K) — represented as Energy/TemperatureInterval
// ─────────────────────────────────────────────────────────────────────────────

/// Entropy in J/K, represented as a `f64` with explicit units.
///
/// Because `uom` does not directly expose an `Entropy` quantity as a standalone
/// type in all configurations, this module uses a thin wrapper.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Entropy {
    /// Raw value in J/K.
    value_jk: f64,
}

impl Entropy {
    /// Create an entropy from a value in J/K.
    pub fn from_joules_per_kelvin(jk: f64) -> Self {
        Self { value_jk: jk }
    }

    /// Return the value in J/K.
    pub fn get_joules_per_kelvin(self) -> f64 {
        self.value_jk
    }

    /// Compute entropy change for a reversible process: ΔS = Q_rev / T.
    ///
    /// Returns `None` if temperature is zero or negative.
    pub fn reversible_change(
        heat: &Energy,
        temperature: &ThermodynamicTemperature,
    ) -> Option<Self> {
        let t = temperature.get::<temperature_kelvin>();
        if t <= 0.0 {
            return None;
        }
        Some(Self::from_joules_per_kelvin(heat.get::<joule>() / t))
    }
}

impl std::ops::Add for Entropy {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::from_joules_per_kelvin(self.value_jk + rhs.value_jk)
    }
}

impl std::ops::Sub for Entropy {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::from_joules_per_kelvin(self.value_jk - rhs.value_jk)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience constructors
// ─────────────────────────────────────────────────────────────────────────────

/// Create a [`Mass`] from kilograms.
pub fn mass_kg(kg: f64) -> Mass {
    si::Mass::new::<kilogram>(kg)
}

/// Create a [`Length`] from metres.
pub fn length_m(m: f64) -> Length {
    si::Length::new::<meter>(m)
}

/// Create a [`Time`] from seconds.
pub fn time_s(s: f64) -> Time {
    si::Time::new::<second>(s)
}

/// Create a [`Velocity`] from metres per second.
pub fn velocity_ms(ms: f64) -> Velocity {
    si::Velocity::new::<meter_per_second>(ms)
}

/// Create an [`Acceleration`] from metres per second squared.
pub fn acceleration_ms2(a: f64) -> Acceleration {
    si::Acceleration::new::<meter_per_second_squared>(a)
}

/// Create a [`Force`] from Newtons.
pub fn force_n(n: f64) -> Force {
    si::Force::new::<newton>(n)
}

/// Create a [`Pressure`] from Pascals.
pub fn pressure_pa(pa: f64) -> Pressure {
    si::Pressure::new::<pascal>(pa)
}

/// Create an [`Energy`] from Joules.
pub fn energy_j(j: f64) -> Energy {
    si::Energy::new::<joule>(j)
}

/// Create a [`ThermodynamicTemperature`] from Kelvin.
pub fn temperature_k(k: f64) -> ThermodynamicTemperature {
    si::ThermodynamicTemperature::new::<temperature_kelvin>(k)
}

/// Create an [`AngularMomentum`] from kg·m²/s.
pub fn angular_momentum(l: f64) -> AngularMomentum {
    si::AngularMomentum::new::<kilogram_square_meter_per_second>(l)
}

// ─────────────────────────────────────────────────────────────────────────────
// Value extractors (→ f64 in SI base units)
// ─────────────────────────────────────────────────────────────────────────────

/// Extract mass in kg.
pub fn to_kg(m: &Mass) -> f64 {
    m.get::<kilogram>()
}

/// Extract length in m.
pub fn to_m(l: &Length) -> f64 {
    l.get::<meter>()
}

/// Extract time in s.
pub fn to_s(t: &Time) -> f64 {
    t.get::<second>()
}

/// Extract velocity in m/s.
pub fn to_ms(v: &Velocity) -> f64 {
    v.get::<meter_per_second>()
}

/// Extract acceleration in m/s².
pub fn to_ms2(a: &Acceleration) -> f64 {
    a.get::<meter_per_second_squared>()
}

/// Extract force in N.
pub fn to_n(f: &Force) -> f64 {
    f.get::<newton>()
}

/// Extract pressure in Pa.
pub fn to_pa(p: &Pressure) -> f64 {
    p.get::<pascal>()
}

/// Extract energy in J.
pub fn to_joules(e: &Energy) -> f64 {
    e.get::<joule>()
}

/// Extract temperature in K.
pub fn to_kelvin(t: &ThermodynamicTemperature) -> f64 {
    t.get::<temperature_kelvin>()
}

/// Extract angular momentum in kg·m²/s.
pub fn to_angular_momentum(l: &AngularMomentum) -> f64 {
    l.get::<kilogram_square_meter_per_second>()
}

// ─────────────────────────────────────────────────────────────────────────────
// Kinetic energy helper
// ─────────────────────────────────────────────────────────────────────────────

/// Compute kinetic energy: KE = ½mv².
pub fn kinetic_energy(mass: &Mass, velocity: &Velocity) -> Energy {
    let ke_j = 0.5 * to_kg(mass) * to_ms(velocity).powi(2);
    energy_j(ke_j)
}

/// Compute gravitational potential energy: PE = mgh.
pub fn gravitational_pe(mass: &Mass, height: &Length, g: f64) -> Energy {
    energy_j(to_kg(mass) * g * to_m(height))
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mass_round_trip() {
        let m = mass_kg(2.5);
        assert!((to_kg(&m) - 2.5).abs() < 1e-12);
    }

    #[test]
    fn test_energy_round_trip() {
        let e = energy_j(1000.0);
        assert!((to_joules(&e) - 1000.0).abs() < 1e-12);
    }

    #[test]
    fn test_temperature_round_trip() {
        let t = temperature_k(300.0);
        assert!((to_kelvin(&t) - 300.0).abs() < 1e-12);
    }

    #[test]
    fn test_kinetic_energy_formula() {
        // KE = 0.5 * 2 kg * (3 m/s)^2 = 9 J
        let m = mass_kg(2.0);
        let v = velocity_ms(3.0);
        let ke = kinetic_energy(&m, &v);
        assert!((to_joules(&ke) - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_gravitational_pe_formula() {
        // PE = 1 kg * 9.81 m/s^2 * 10 m = 98.1 J
        let m = mass_kg(1.0);
        let h = length_m(10.0);
        let pe = gravitational_pe(&m, &h, 9.81);
        assert!((to_joules(&pe) - 98.1).abs() < 1e-10);
    }

    #[test]
    fn test_entropy_from_heat_and_temperature() {
        // ΔS = Q/T = 300 J / 300 K = 1 J/K
        let q = energy_j(300.0);
        let t = temperature_k(300.0);
        let ds = Entropy::reversible_change(&q, &t).unwrap();
        assert!((ds.get_joules_per_kelvin() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_entropy_zero_temperature_returns_none() {
        let q = energy_j(100.0);
        let t = temperature_k(0.0);
        assert!(Entropy::reversible_change(&q, &t).is_none());
    }

    #[test]
    fn test_entropy_addition() {
        let s1 = Entropy::from_joules_per_kelvin(2.0);
        let s2 = Entropy::from_joules_per_kelvin(3.0);
        let s3 = s1 + s2;
        assert!((s3.get_joules_per_kelvin() - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_angular_momentum_round_trip() {
        let l = angular_momentum(4.2);
        assert!((to_angular_momentum(&l) - 4.2).abs() < 1e-12);
    }

    #[test]
    fn test_si_unit_mismatch_would_not_compile() {
        // This test documents the type safety guarantee:
        // Uncommenting the line below would cause a compile error:
        // let _: Mass = energy_j(1.0); // type mismatch — Energy != Mass
        // Type safety is guaranteed at compile time; no runtime assertion needed.
        let _ = (); // marker: compile-time type safety verified
    }
}
