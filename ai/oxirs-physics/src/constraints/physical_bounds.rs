//! Physical Bounds Validation
//!
//! Validates simulation parameters and results against known physical limits

use crate::error::{PhysicsError, PhysicsResult};
use crate::simulation::parameter_extraction::{PhysicalQuantity, SimulationParameters};
use crate::simulation::result_injection::{SimulationResult, StateVector};

/// Physical bounds for various quantities
pub struct PhysicalBounds;

impl PhysicalBounds {
    /// Validate temperature bounds
    ///
    /// - Absolute zero: T >= 0 K
    /// - Practical upper limit: T < 10^9 K (stellar core temperatures)
    pub fn validate_temperature(temp_kelvin: f64) -> PhysicsResult<()> {
        const ABSOLUTE_ZERO: f64 = 0.0;
        const MAX_REALISTIC_TEMP: f64 = 1e9; // 1 billion K (stellar cores)

        if temp_kelvin < ABSOLUTE_ZERO {
            return Err(PhysicsError::ConstraintViolation(format!(
                "Temperature {} K violates absolute zero (T >= 0 K)",
                temp_kelvin
            )));
        }

        if temp_kelvin > MAX_REALISTIC_TEMP {
            return Err(PhysicsError::ConstraintViolation(format!(
                "Temperature {} K exceeds realistic limit ({} K)",
                temp_kelvin, MAX_REALISTIC_TEMP
            )));
        }

        Ok(())
    }

    /// Validate pressure bounds
    ///
    /// - Vacuum: P >= 0 Pa
    /// - Practical upper limit: P < 10^15 Pa (Earth's core pressure)
    pub fn validate_pressure(pressure_pa: f64) -> PhysicsResult<()> {
        const MIN_PRESSURE: f64 = 0.0;
        const MAX_REALISTIC_PRESSURE: f64 = 1e15; // Earth's core

        if pressure_pa < MIN_PRESSURE {
            return Err(PhysicsError::ConstraintViolation(format!(
                "Pressure {} Pa is negative (P >= 0 for real fluids)",
                pressure_pa
            )));
        }

        if pressure_pa > MAX_REALISTIC_PRESSURE {
            return Err(PhysicsError::ConstraintViolation(format!(
                "Pressure {} Pa exceeds realistic limit ({} Pa)",
                pressure_pa, MAX_REALISTIC_PRESSURE
            )));
        }

        Ok(())
    }

    /// Validate velocity bounds (subluminal)
    ///
    /// - Speed of light: v < c = 299,792,458 m/s
    pub fn validate_velocity(velocity_ms: f64) -> PhysicsResult<()> {
        const SPEED_OF_LIGHT: f64 = 299_792_458.0; // m/s

        if velocity_ms.abs() >= SPEED_OF_LIGHT {
            return Err(PhysicsError::ConstraintViolation(format!(
                "Velocity {} m/s exceeds speed of light ({} m/s)",
                velocity_ms, SPEED_OF_LIGHT
            )));
        }

        Ok(())
    }

    /// Validate density bounds
    ///
    /// - Positive: ρ > 0 kg/m³
    /// - Practical upper limit: ρ < 10^18 kg/m³ (neutron star density)
    pub fn validate_density(density_kgm3: f64) -> PhysicsResult<()> {
        const MIN_DENSITY: f64 = 1e-10; // Near-vacuum
        const MAX_REALISTIC_DENSITY: f64 = 1e18; // Neutron star

        if density_kgm3 <= 0.0 {
            return Err(PhysicsError::ConstraintViolation(format!(
                "Density {} kg/m³ must be positive",
                density_kgm3
            )));
        }

        if density_kgm3 < MIN_DENSITY {
            tracing::warn!(
                "Density {} kg/m³ is extremely low (near-vacuum)",
                density_kgm3
            );
        }

        if density_kgm3 > MAX_REALISTIC_DENSITY {
            return Err(PhysicsError::ConstraintViolation(format!(
                "Density {} kg/m³ exceeds realistic limit ({} kg/m³)",
                density_kgm3, MAX_REALISTIC_DENSITY
            )));
        }

        Ok(())
    }

    /// Validate stress against material yield stress (von Mises criterion)
    pub fn validate_stress(stress_pa: f64, yield_stress_pa: f64) -> PhysicsResult<()> {
        if stress_pa > yield_stress_pa {
            return Err(PhysicsError::ConstraintViolation(format!(
                "Stress {} Pa exceeds yield stress {} Pa (material will yield)",
                stress_pa, yield_stress_pa
            )));
        }

        Ok(())
    }

    /// Validate mass (positive and non-zero)
    pub fn validate_mass(mass_kg: f64) -> PhysicsResult<()> {
        if mass_kg <= 0.0 {
            return Err(PhysicsError::ConstraintViolation(format!(
                "Mass {} kg must be positive",
                mass_kg
            )));
        }

        Ok(())
    }

    /// Validate time (non-negative)
    pub fn validate_time(time_s: f64) -> PhysicsResult<()> {
        if time_s < 0.0 {
            return Err(PhysicsError::ConstraintViolation(format!(
                "Time {} s cannot be negative",
                time_s
            )));
        }

        Ok(())
    }
}

/// Bounds Validator - validates parameters and results
pub struct BoundsValidator;

impl BoundsValidator {
    pub fn new() -> Self {
        Self
    }

    /// Validate simulation parameters before running
    pub fn validate_parameters(&self, params: &SimulationParameters) -> PhysicsResult<()> {
        // Validate time span
        PhysicalBounds::validate_time(params.time_span.0)?;
        PhysicalBounds::validate_time(params.time_span.1)?;

        if params.time_span.1 <= params.time_span.0 {
            return Err(PhysicsError::ConstraintViolation(
                "End time must be greater than start time".to_string(),
            ));
        }

        if params.time_steps == 0 {
            return Err(PhysicsError::ConstraintViolation(
                "Time steps must be greater than zero".to_string(),
            ));
        }

        // Validate initial conditions based on simulation type
        match params.simulation_type.as_str() {
            "thermal" => self.validate_thermal_parameters(params)?,
            "mechanical" => self.validate_mechanical_parameters(params)?,
            "fluid" => self.validate_fluid_parameters(params)?,
            _ => {} // Unknown type, skip specific validation
        }

        Ok(())
    }

    /// Validate thermal simulation parameters
    fn validate_thermal_parameters(&self, params: &SimulationParameters) -> PhysicsResult<()> {
        // Validate initial temperature
        if let Some(temp) = params.initial_conditions.get("temperature") {
            let temp_k = self.convert_to_kelvin(temp)?;
            PhysicalBounds::validate_temperature(temp_k)?;
        }

        // Validate material properties
        if let Some(density) = params.material_properties.get("density") {
            PhysicalBounds::validate_density(density.value)?;
        }

        Ok(())
    }

    /// Validate mechanical simulation parameters
    fn validate_mechanical_parameters(&self, params: &SimulationParameters) -> PhysicsResult<()> {
        // Validate material properties
        if let Some(density) = params.material_properties.get("density") {
            PhysicalBounds::validate_density(density.value)?;
        }

        // Validate Young's modulus (must be positive)
        if let Some(e) = params.material_properties.get("youngs_modulus") {
            if e.value <= 0.0 {
                return Err(PhysicsError::ConstraintViolation(
                    "Young's modulus must be positive".to_string(),
                ));
            }
        }

        // Validate Poisson's ratio (-1 < ν < 0.5 for isotropic materials)
        if let Some(nu) = params.material_properties.get("poisson_ratio") {
            if nu.value <= -1.0 || nu.value >= 0.5 {
                return Err(PhysicsError::ConstraintViolation(format!(
                    "Poisson's ratio {} must be in range (-1, 0.5)",
                    nu.value
                )));
            }
        }

        Ok(())
    }

    /// Validate fluid simulation parameters
    fn validate_fluid_parameters(&self, params: &SimulationParameters) -> PhysicsResult<()> {
        // Validate density
        if let Some(density) = params.material_properties.get("density") {
            PhysicalBounds::validate_density(density.value)?;
        }

        // Validate viscosity (must be non-negative)
        if let Some(mu) = params.material_properties.get("viscosity") {
            if mu.value < 0.0 {
                return Err(PhysicsError::ConstraintViolation(
                    "Viscosity must be non-negative".to_string(),
                ));
            }
        }

        // Validate initial pressure
        if let Some(p) = params.initial_conditions.get("pressure") {
            PhysicalBounds::validate_pressure(p.value)?;
        }

        // Validate initial velocity
        if let Some(v) = params.initial_conditions.get("velocity") {
            PhysicalBounds::validate_velocity(v.value)?;
        }

        Ok(())
    }

    /// Validate simulation results after computation
    pub fn validate_results(&self, result: &SimulationResult) -> PhysicsResult<()> {
        // Validate all state vectors
        for state in &result.state_trajectory {
            self.validate_state_vector(state)?;
        }

        // Check convergence
        if !result.convergence_info.converged {
            tracing::warn!(
                "Simulation did not converge (residual: {})",
                result.convergence_info.final_residual
            );
        }

        Ok(())
    }

    /// Validate a single state vector
    fn validate_state_vector(&self, state: &StateVector) -> PhysicsResult<()> {
        // Validate time
        PhysicalBounds::validate_time(state.time)?;

        // Validate state variables
        for (key, value) in &state.state {
            match key.as_str() {
                "temperature" => PhysicalBounds::validate_temperature(*value)?,
                "pressure" => PhysicalBounds::validate_pressure(*value)?,
                "velocity" => PhysicalBounds::validate_velocity(*value)?,
                "density" => PhysicalBounds::validate_density(*value)?,
                "mass" => PhysicalBounds::validate_mass(*value)?,
                _ => {} // Unknown variable, skip validation
            }
        }

        Ok(())
    }

    /// Convert temperature to Kelvin
    fn convert_to_kelvin(&self, temp: &PhysicalQuantity) -> PhysicsResult<f64> {
        match temp.unit.as_str() {
            "K" => Ok(temp.value),
            "°C" | "C" => Ok(temp.value + 273.15),
            "°F" | "F" => Ok((temp.value - 32.0) * 5.0 / 9.0 + 273.15),
            _ => Err(PhysicsError::UnitConversion(format!(
                "Unknown temperature unit: {}",
                temp.unit
            ))),
        }
    }
}

impl Default for BoundsValidator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_temperature_bounds() {
        // Valid temperatures
        assert!(PhysicalBounds::validate_temperature(0.0).is_ok()); // Absolute zero
        assert!(PhysicalBounds::validate_temperature(293.15).is_ok()); // Room temp
        assert!(PhysicalBounds::validate_temperature(5778.0).is_ok()); // Sun surface

        // Invalid temperatures
        assert!(PhysicalBounds::validate_temperature(-1.0).is_err()); // Below absolute zero
        assert!(PhysicalBounds::validate_temperature(1e10).is_err()); // Too hot
    }

    #[test]
    fn test_pressure_bounds() {
        // Valid pressures
        assert!(PhysicalBounds::validate_pressure(0.0).is_ok()); // Vacuum
        assert!(PhysicalBounds::validate_pressure(101325.0).is_ok()); // 1 atm
        assert!(PhysicalBounds::validate_pressure(1e6).is_ok()); // 10 atm

        // Invalid pressures
        assert!(PhysicalBounds::validate_pressure(-1.0).is_err()); // Negative
        assert!(PhysicalBounds::validate_pressure(1e16).is_err()); // Too high
    }

    #[test]
    fn test_velocity_bounds() {
        // Valid velocities
        assert!(PhysicalBounds::validate_velocity(0.0).is_ok());
        assert!(PhysicalBounds::validate_velocity(340.0).is_ok()); // Speed of sound
        assert!(PhysicalBounds::validate_velocity(11_200.0).is_ok()); // Escape velocity

        // Invalid velocities
        assert!(PhysicalBounds::validate_velocity(3e8).is_err()); // Faster than light
        assert!(PhysicalBounds::validate_velocity(-3e8).is_err()); // Faster than light (negative)
    }

    #[test]
    fn test_density_bounds() {
        // Valid densities
        assert!(PhysicalBounds::validate_density(1.0).is_ok()); // Water
        assert!(PhysicalBounds::validate_density(1000.0).is_ok()); // Water (kg/m³)
        assert!(PhysicalBounds::validate_density(7850.0).is_ok()); // Steel

        // Invalid densities
        assert!(PhysicalBounds::validate_density(0.0).is_err()); // Zero
        assert!(PhysicalBounds::validate_density(-1.0).is_err()); // Negative
        assert!(PhysicalBounds::validate_density(1e19).is_err()); // Too high
    }

    #[test]
    fn test_bounds_validator() {
        let validator = BoundsValidator::new();

        let mut ic = HashMap::new();
        ic.insert(
            "temperature".to_string(),
            PhysicalQuantity {
                value: 300.0,
                unit: "K".to_string(),
                uncertainty: None,
            },
        );

        let params = SimulationParameters {
            entity_iri: "urn:test".to_string(),
            simulation_type: "thermal".to_string(),
            initial_conditions: ic,
            boundary_conditions: Vec::new(),
            time_span: (0.0, 100.0),
            time_steps: 10,
            material_properties: HashMap::new(),
            constraints: Vec::new(),
        };

        assert!(validator.validate_parameters(&params).is_ok());
    }

    #[test]
    fn test_invalid_parameters() {
        let validator = BoundsValidator::new();

        // Invalid time span
        let params = SimulationParameters {
            entity_iri: "urn:test".to_string(),
            simulation_type: "thermal".to_string(),
            initial_conditions: HashMap::new(),
            boundary_conditions: Vec::new(),
            time_span: (100.0, 0.0), // End before start
            time_steps: 10,
            material_properties: HashMap::new(),
            constraints: Vec::new(),
        };

        assert!(validator.validate_parameters(&params).is_err());

        // Zero time steps
        let params = SimulationParameters {
            entity_iri: "urn:test".to_string(),
            simulation_type: "thermal".to_string(),
            initial_conditions: HashMap::new(),
            boundary_conditions: Vec::new(),
            time_span: (0.0, 100.0),
            time_steps: 0, // Invalid
            material_properties: HashMap::new(),
            constraints: Vec::new(),
        };

        assert!(validator.validate_parameters(&params).is_err());
    }

    #[test]
    fn test_temperature_conversion() {
        let validator = BoundsValidator::new();

        // Kelvin
        let temp_k = PhysicalQuantity {
            value: 300.0,
            unit: "K".to_string(),
            uncertainty: None,
        };
        assert!((validator.convert_to_kelvin(&temp_k).unwrap() - 300.0).abs() < 1e-6);

        // Celsius
        let temp_c = PhysicalQuantity {
            value: 0.0,
            unit: "°C".to_string(),
            uncertainty: None,
        };
        assert!((validator.convert_to_kelvin(&temp_c).unwrap() - 273.15).abs() < 1e-6);

        // Fahrenheit
        let temp_f = PhysicalQuantity {
            value: 32.0,
            unit: "°F".to_string(),
            uncertainty: None,
        };
        assert!((validator.convert_to_kelvin(&temp_f).unwrap() - 273.15).abs() < 1e-6);
    }
}
