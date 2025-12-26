//! SciRS2-based Thermal Simulation
//!
//! Real thermal simulation using scirs2-integrate ODE solvers

use super::parameter_extraction::SimulationParameters;
use super::result_injection::{
    ConvergenceInfo, SimulationProvenance, SimulationResult, StateVector,
};
use super::simulation_runner::PhysicsSimulation;
use crate::error::{PhysicsError, PhysicsResult};
use async_trait::async_trait;
use chrono::Utc;
use std::collections::HashMap;
use uuid::Uuid;

/// Thermal Simulation using SciRS2
pub struct SciRS2ThermalSimulation {
    /// Thermal conductivity (W/m·K)
    #[cfg_attr(not(feature = "simulation"), allow(dead_code))]
    conductivity: f64,

    /// Specific heat capacity (J/kg·K)
    #[cfg_attr(not(feature = "simulation"), allow(dead_code))]
    specific_heat: f64,

    /// Density (kg/m³)
    #[cfg_attr(not(feature = "simulation"), allow(dead_code))]
    density: f64,
}

impl Default for SciRS2ThermalSimulation {
    fn default() -> Self {
        Self {
            conductivity: 1.0,     // Default: water-like
            specific_heat: 4186.0, // Water: 4186 J/kg·K
            density: 1000.0,       // Water: 1000 kg/m³
        }
    }
}

impl SciRS2ThermalSimulation {
    pub fn new(conductivity: f64, specific_heat: f64, density: f64) -> Self {
        Self {
            conductivity,
            specific_heat,
            density,
        }
    }
}

#[async_trait]
impl PhysicsSimulation for SciRS2ThermalSimulation {
    fn simulation_type(&self) -> &str {
        "thermal"
    }

    async fn run(&self, params: &SimulationParameters) -> PhysicsResult<SimulationResult> {
        #[cfg(feature = "simulation")]
        {
            self.run_with_scirs2(params).await
        }

        #[cfg(not(feature = "simulation"))]
        {
            // Fallback to mock simulation
            self.run_mock(params).await
        }
    }

    fn validate_results(&self, result: &SimulationResult) -> PhysicsResult<()> {
        if !result.convergence_info.converged {
            return Err(PhysicsError::Simulation(
                "Thermal simulation did not converge".to_string(),
            ));
        }

        // Check temperature bounds (physical constraints)
        for state in &result.state_trajectory {
            if let Some(&temp) = state.state.get("temperature") {
                if !(0.0..=1000.0).contains(&temp) {
                    return Err(PhysicsError::ConstraintViolation(format!(
                        "Temperature {} K out of physical bounds [0, 1000]",
                        temp
                    )));
                }
            }
        }

        Ok(())
    }
}

impl SciRS2ThermalSimulation {
    #[cfg(feature = "simulation")]
    async fn run_with_scirs2(
        &self,
        params: &SimulationParameters,
    ) -> PhysicsResult<SimulationResult> {
        use scirs2_core::ndarray_ext::{Array1, ArrayView1};
        use scirs2_integrate::ode::{solve_ivp, ODEMethod, ODEOptions};

        // Extract initial temperature from parameters
        let initial_temp = params
            .initial_conditions
            .get("temperature")
            .map(|q| q.value)
            .unwrap_or(293.15); // Default: 20°C = 293.15 K

        // Setup initial state (1D temperature distribution)
        let n_points = 10;
        let initial_state: Array1<f64> = Array1::from_elem(n_points, initial_temp);

        // Thermal diffusivity
        let alpha = self.conductivity / (self.density * self.specific_heat);
        let dx = 0.1; // Grid spacing

        // Define the thermal ODE: dT/dt = α ∇²T
        let thermal_ode = move |_t: f64, y: ArrayView1<f64>| -> Array1<f64> {
            let mut derivatives = Array1::zeros(n_points);
            for i in 0..n_points {
                let left = if i > 0 { y[i - 1] } else { y[i] };
                let right = if i < n_points - 1 { y[i + 1] } else { y[i] };
                let center = y[i];
                derivatives[i] = alpha * (left - 2.0 * center + right) / (dx * dx);
            }
            derivatives
        };

        // Solve with RK45
        let options = ODEOptions {
            method: ODEMethod::RK45,
            rtol: 1e-6,
            atol: 1e-8,
            max_step: Some((params.time_span.1 - params.time_span.0) / (params.time_steps as f64)),
            ..Default::default()
        };

        let start_time = std::time::Instant::now();
        let solution = solve_ivp(
            thermal_ode,
            [params.time_span.0, params.time_span.1],
            initial_state,
            Some(options),
        )
        .map_err(|e| PhysicsError::Simulation(format!("SciRS2 ODE solver failed: {:?}", e)))?;
        let elapsed_ms = start_time.elapsed().as_millis() as u64;

        // Resample the solution to match requested time_steps
        // The ODE solver uses adaptive steps, so we interpolate to get the requested number of points
        let mut trajectory = Vec::new();
        let dt = (params.time_span.1 - params.time_span.0) / (params.time_steps as f64);

        for i in 0..params.time_steps {
            let target_time = params.time_span.0 + dt * (i as f64);

            // Find the temperature at this time by linear interpolation
            let temp = interpolate_at_time(&solution.t, &solution.y, target_time, n_points / 2);

            let mut state = HashMap::new();
            state.insert("temperature".to_string(), temp);
            trajectory.push(StateVector {
                time: target_time,
                state,
            });
        }

        Ok(SimulationResult {
            entity_iri: params.entity_iri.clone(),
            simulation_run_id: Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            state_trajectory: trajectory,
            derived_quantities: HashMap::new(),
            convergence_info: ConvergenceInfo {
                converged: solution.success,
                iterations: solution.t.len(),
                final_residual: 1e-6,
            },
            provenance: SimulationProvenance {
                software: "oxirs-physics (SciRS2)".to_string(),
                version: crate::VERSION.to_string(),
                parameters_hash: self.compute_params_hash(params),
                executed_at: Utc::now(),
                execution_time_ms: elapsed_ms,
            },
        })
    }

    #[cfg(not(feature = "simulation"))]
    async fn run_mock(&self, params: &SimulationParameters) -> PhysicsResult<SimulationResult> {
        // Mock simulation (fallback when scirs2-integrate not available)
        let mut trajectory = Vec::new();

        let initial_temp = params
            .initial_conditions
            .get("temperature")
            .map(|q| q.value)
            .unwrap_or(293.15);

        for i in 0..params.time_steps {
            let time = params.time_span.0
                + (params.time_span.1 - params.time_span.0) * (i as f64 / params.time_steps as f64);

            let mut state = HashMap::new();
            state.insert("temperature".to_string(), initial_temp + time * 0.01);

            trajectory.push(StateVector { time, state });
        }

        Ok(SimulationResult {
            entity_iri: params.entity_iri.clone(),
            simulation_run_id: Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            state_trajectory: trajectory,
            derived_quantities: HashMap::new(),
            convergence_info: ConvergenceInfo {
                converged: true,
                iterations: params.time_steps,
                final_residual: 1e-6,
            },
            provenance: SimulationProvenance {
                software: "oxirs-physics (mock)".to_string(),
                version: crate::VERSION.to_string(),
                parameters_hash: self.compute_params_hash(params),
                executed_at: Utc::now(),
                execution_time_ms: 10,
            },
        })
    }

    fn compute_params_hash(&self, params: &SimulationParameters) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        params.entity_iri.hash(&mut hasher);
        params.simulation_type.hash(&mut hasher);
        hasher.finish().to_string()
    }
}

/// Interpolate value at a given time from ODE solver output
/// Uses linear interpolation between the nearest time points
#[cfg(feature = "simulation")]
fn interpolate_at_time(
    times: &[f64],
    values: &[scirs2_core::ndarray_ext::Array1<f64>],
    target_time: f64,
    index: usize,
) -> f64 {
    // Find the two closest time points
    let mut i = 0;
    while i < times.len() - 1 && times[i + 1] < target_time {
        i += 1;
    }

    if i >= times.len() - 1 {
        // Beyond the last point, return last value
        return values[times.len() - 1][index];
    }

    let t0 = times[i];
    let t1 = times[i + 1];

    if (t1 - t0).abs() < 1e-12 {
        return values[i][index];
    }

    // Linear interpolation
    let alpha = (target_time - t0) / (t1 - t0);
    let v0 = values[i][index];
    let v1 = values[i + 1][index];

    v0 + alpha * (v1 - v0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_scirs2_thermal_simulation() {
        let sim = SciRS2ThermalSimulation::default();

        let mut initial_conditions = HashMap::new();
        initial_conditions.insert(
            "temperature".to_string(),
            super::super::parameter_extraction::PhysicalQuantity {
                value: 300.0,
                unit: "K".to_string(),
                uncertainty: None,
            },
        );

        let params = SimulationParameters {
            entity_iri: "urn:example:battery:thermal".to_string(),
            simulation_type: "thermal".to_string(),
            initial_conditions,
            boundary_conditions: Vec::new(),
            time_span: (0.0, 100.0),
            time_steps: 20,
            material_properties: HashMap::new(),
            constraints: Vec::new(),
        };

        let result = sim.run(&params).await.unwrap();

        assert_eq!(result.state_trajectory.len(), 20);
        assert!(result.convergence_info.converged);
        assert!(sim.validate_results(&result).is_ok());
    }
}
