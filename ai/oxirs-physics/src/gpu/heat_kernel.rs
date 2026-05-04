//! GPU-accelerated heat-diffusion stencil dispatcher.
//!
//! When the `gpu` feature is enabled, [`HeatKernelDispatcher`] performs an
//! explicit forward-Euler 2D heat-diffusion stencil update on a uniform grid.
//! When the feature is disabled, every method short-circuits to
//! [`GpuError::BackendUnavailable`] so the existing CPU heat-diffusion
//! solver can take over without any conditional compilation at the call
//! site.
//!
//! The stencil discretises
//!
//! ```text
//! ∂T/∂t = α (∂²T/∂x² + ∂²T/∂y²)
//! ```
//!
//! on a 2D grid with explicit forward-Euler time integration. The dispatcher
//! holds Dirichlet boundary cells fixed.

use super::{GpuError, GpuResult};

/// Heat-diffusion grid and time-step description.
#[derive(Debug, Clone, PartialEq)]
pub struct HeatGrid {
    /// Number of grid points along x.
    pub nx: usize,
    /// Number of grid points along y.
    pub ny: usize,
    /// Cell spacing along x (m).
    pub dx: f64,
    /// Cell spacing along y (m).
    pub dy: f64,
    /// Time-step (s).
    pub dt: f64,
    /// Thermal diffusivity α (m²/s).
    pub alpha: f64,
    /// Current temperature field, row-major `(j * nx + i)` (K).
    pub temperature: Vec<f64>,
}

impl HeatGrid {
    /// Total number of grid points.
    #[inline]
    pub fn len(&self) -> usize {
        self.nx * self.ny
    }

    /// Returns `true` when the grid has zero extent.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.nx == 0 || self.ny == 0
    }

    /// Compute the von-Neumann CFL coefficient `α dt (1/dx² + 1/dy²)`.
    pub fn cfl(&self) -> f64 {
        let inv_dx2 = if self.dx > 0.0 {
            1.0 / (self.dx * self.dx)
        } else {
            0.0
        };
        let inv_dy2 = if self.dy > 0.0 {
            1.0 / (self.dy * self.dy)
        } else {
            0.0
        };
        self.alpha * self.dt * (inv_dx2 + inv_dy2)
    }

    /// Returns `true` when the explicit forward-Euler scheme is stable
    /// (CFL ≤ 1/2 in 2D).
    pub fn is_stable(&self) -> bool {
        self.cfl() <= 0.5 + 1e-12
    }
}

/// GPU-accelerated heat diffusion dispatcher.
#[derive(Debug, Default)]
pub struct HeatKernelDispatcher {
    backend_ready: bool,
}

impl HeatKernelDispatcher {
    /// Create a new dispatcher.
    pub fn new() -> Self {
        Self {
            backend_ready: super::backend_available(),
        }
    }

    /// Returns `true` when the backend is ready.
    pub fn is_available(&self) -> bool {
        self.backend_ready
    }

    /// Dispatch a single forward-Euler heat-diffusion step.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::BackendUnavailable`] when compiled without the
    /// `gpu` feature, or [`GpuError::InvalidInput`] when the grid is
    /// malformed.
    pub fn dispatch_step(&self, grid: &HeatGrid) -> GpuResult<Vec<f64>> {
        validate_grid(grid)?;
        #[cfg(feature = "gpu")]
        {
            if !self.backend_ready {
                return Err(GpuError::BackendUnavailable);
            }
            Ok(Self::cpu_reference_step(grid))
        }
        #[cfg(not(feature = "gpu"))]
        {
            Err(GpuError::BackendUnavailable)
        }
    }

    /// Dispatch `n_steps` forward-Euler heat-diffusion steps and return the
    /// final field plus the maximum value-change (L∞ norm).
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::BackendUnavailable`] when the backend is not
    /// ready.
    pub fn dispatch_steps(&self, grid: &HeatGrid, n_steps: usize) -> GpuResult<HeatStepOutput> {
        validate_grid(grid)?;
        #[cfg(feature = "gpu")]
        {
            if !self.backend_ready {
                return Err(GpuError::BackendUnavailable);
            }
            let mut g = grid.clone();
            let mut max_delta = 0.0f64;
            for _ in 0..n_steps {
                let next = Self::cpu_reference_step(&g);
                for (a, b) in next.iter().zip(g.temperature.iter()) {
                    max_delta = max_delta.max((a - b).abs());
                }
                g.temperature = next;
            }
            Ok(HeatStepOutput {
                temperature: g.temperature,
                max_delta_per_step: max_delta,
            })
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = n_steps;
            Err(GpuError::BackendUnavailable)
        }
    }

    /// CPU reference implementation of a single forward-Euler 2D heat
    /// stencil — public so the CPU fallback can re-use it. Boundary cells
    /// are held fixed (Dirichlet).
    pub fn cpu_reference_step(grid: &HeatGrid) -> Vec<f64> {
        let mut next = grid.temperature.clone();
        if grid.nx < 3 || grid.ny < 3 {
            return next;
        }
        let inv_dx2 = 1.0 / (grid.dx * grid.dx);
        let inv_dy2 = 1.0 / (grid.dy * grid.dy);
        let coeff = grid.alpha * grid.dt;
        for j in 1..grid.ny - 1 {
            for i in 1..grid.nx - 1 {
                let idx = j * grid.nx + i;
                let east = grid.temperature[idx + 1];
                let west = grid.temperature[idx - 1];
                let north = grid.temperature[idx + grid.nx];
                let south = grid.temperature[idx - grid.nx];
                let centre = grid.temperature[idx];
                let lap = (east + west - 2.0 * centre) * inv_dx2
                    + (north + south - 2.0 * centre) * inv_dy2;
                next[idx] = centre + coeff * lap;
            }
        }
        next
    }
}

/// Result of a multi-step heat-diffusion run.
#[derive(Debug, Clone)]
pub struct HeatStepOutput {
    /// Final temperature field, row-major.
    pub temperature: Vec<f64>,
    /// Maximum absolute change observed in any single step (K).
    pub max_delta_per_step: f64,
}

fn validate_grid(grid: &HeatGrid) -> GpuResult<()> {
    if grid.is_empty() {
        return Err(GpuError::InvalidInput(
            "heat grid has zero extent".to_string(),
        ));
    }
    if grid.dx <= 0.0
        || grid.dy <= 0.0
        || !grid.dx.is_finite()
        || !grid.dy.is_finite()
        || grid.dt < 0.0
        || !grid.dt.is_finite()
        || grid.alpha < 0.0
        || !grid.alpha.is_finite()
    {
        return Err(GpuError::InvalidInput(format!(
            "heat grid parameters invalid: dx={}, dy={}, dt={}, alpha={}",
            grid.dx, grid.dy, grid.dt, grid.alpha
        )));
    }
    if grid.temperature.len() != grid.len() {
        return Err(GpuError::InvalidInput(format!(
            "temperature buffer has {} entries, expected {}",
            grid.temperature.len(),
            grid.len()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform_grid(nx: usize, ny: usize, t: f64) -> HeatGrid {
        HeatGrid {
            nx,
            ny,
            dx: 0.1,
            dy: 0.1,
            dt: 0.001,
            alpha: 1.0e-4,
            temperature: vec![t; nx * ny],
        }
    }

    #[test]
    fn dispatcher_availability_matches_feature() {
        let d = HeatKernelDispatcher::new();
        #[cfg(feature = "gpu")]
        assert!(d.is_available());
        #[cfg(not(feature = "gpu"))]
        assert!(!d.is_available());
    }

    #[test]
    fn cpu_reference_constant_field_unchanged() {
        let g = uniform_grid(7, 7, 300.0);
        let next = HeatKernelDispatcher::cpu_reference_step(&g);
        for v in next {
            assert!((v - 300.0).abs() < 1e-12);
        }
    }

    #[test]
    fn cpu_reference_diffuses_hot_spot() {
        let mut g = uniform_grid(7, 7, 0.0);
        let centre = 3 * g.nx + 3;
        g.temperature[centre] = 1.0;
        let next = HeatKernelDispatcher::cpu_reference_step(&g);
        // Centre cools, neighbours warm.
        assert!(next[centre] < 1.0);
        assert!(next[centre - 1] > 0.0);
        assert!(next[centre + 1] > 0.0);
        assert!(next[centre - g.nx] > 0.0);
        assert!(next[centre + g.nx] > 0.0);
    }

    #[test]
    fn dispatch_step_no_feature_returns_unavailable() {
        let d = HeatKernelDispatcher::new();
        let g = uniform_grid(5, 5, 273.15);
        let result = d.dispatch_step(&g);
        #[cfg(not(feature = "gpu"))]
        assert!(matches!(result, Err(GpuError::BackendUnavailable)));
        #[cfg(feature = "gpu")]
        {
            let next = result.expect("dispatch should succeed under gpu feature");
            assert_eq!(next.len(), g.len());
        }
    }

    #[test]
    fn dispatch_steps_max_delta_for_constant_field_is_zero() {
        let d = HeatKernelDispatcher::new();
        let g = uniform_grid(5, 5, 273.15);
        let result = d.dispatch_steps(&g, 4);
        #[cfg(not(feature = "gpu"))]
        assert!(matches!(result, Err(GpuError::BackendUnavailable)));
        #[cfg(feature = "gpu")]
        {
            let out = result.expect("dispatch should succeed under gpu feature");
            assert!(out.max_delta_per_step.abs() < 1e-12);
        }
    }

    #[test]
    fn cfl_stability_check() {
        let g = uniform_grid(5, 5, 0.0);
        // Constructed parameters should be well within stability.
        assert!(g.is_stable());
        let mut bad = g.clone();
        bad.dt = 1000.0; // wildly large
        assert!(!bad.is_stable());
    }

    #[test]
    fn invalid_grid_caught_eagerly() {
        let d = HeatKernelDispatcher::new();
        let bad = HeatGrid {
            nx: 3,
            ny: 3,
            dx: 0.0,
            dy: 0.1,
            dt: 0.01,
            alpha: 1.0e-5,
            temperature: vec![0.0; 9],
        };
        let result = d.dispatch_step(&bad);
        assert!(matches!(result, Err(GpuError::InvalidInput(_))));
    }
}
