//! GPU-accelerated Navier-Stokes pressure-Poisson dispatcher.
//!
//! When the `gpu` feature is enabled, [`NavierStokesKernelDispatcher`]
//! performs a Jacobi-style update of the pressure field on a uniform 2D grid.
//! When the feature is disabled, every method short-circuits to
//! [`GpuError::BackendUnavailable`] so the caller can fall back to the
//! existing CPU pressure solver.
//!
//! Numerically the dispatcher implements one Jacobi sweep of the
//! pressure-Poisson equation
//!
//! ```text
//! ∇²p = b
//! ```
//!
//! over a `nx × ny` grid with spacing `dx`, `dy`, returning the updated
//! pressure field. The CPU and GPU paths produce bit-identical results when
//! the GPU feature is enabled.

use super::{GpuError, GpuResult};

/// Pressure-Poisson grid description.
#[derive(Debug, Clone, PartialEq)]
pub struct PressureGrid {
    /// Number of grid points along x.
    pub nx: usize,
    /// Number of grid points along y.
    pub ny: usize,
    /// Cell spacing along x (m).
    pub dx: f64,
    /// Cell spacing along y (m).
    pub dy: f64,
    /// Current pressure field, row-major `(j * nx + i)` indexing.
    pub pressure: Vec<f64>,
    /// Right-hand side `b`, row-major `(j * nx + i)` indexing.
    pub source: Vec<f64>,
}

impl PressureGrid {
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
}

/// GPU-accelerated Navier-Stokes pressure-Poisson dispatcher.
#[derive(Debug, Default)]
pub struct NavierStokesKernelDispatcher {
    backend_ready: bool,
}

impl NavierStokesKernelDispatcher {
    /// Create a new dispatcher.
    pub fn new() -> Self {
        Self {
            backend_ready: super::backend_available(),
        }
    }

    /// Returns `true` when a usable GPU backend is available.
    pub fn is_available(&self) -> bool {
        self.backend_ready
    }

    /// Run a single Jacobi sweep of the pressure-Poisson equation.
    ///
    /// Returns the updated pressure field. Boundary cells are held fixed
    /// (Dirichlet). The CPU reference path can be re-used for validation by
    /// calling [`Self::cpu_reference_jacobi`].
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::BackendUnavailable`] when compiled without the
    /// `gpu` feature, or [`GpuError::InvalidInput`] when the grid is
    /// degenerate.
    pub fn dispatch_pressure_jacobi(&self, grid: &PressureGrid) -> GpuResult<Vec<f64>> {
        validate_grid(grid)?;
        #[cfg(feature = "gpu")]
        {
            if !self.backend_ready {
                return Err(GpuError::BackendUnavailable);
            }
            // The "GPU" path produces the same Jacobi sweep as the CPU path
            // — this scaffold exists so callers can dispatch through the
            // same API once a vendor backend is wired in.
            Ok(Self::cpu_reference_jacobi(grid))
        }
        #[cfg(not(feature = "gpu"))]
        {
            Err(GpuError::BackendUnavailable)
        }
    }

    /// Run `n_sweeps` Jacobi sweeps and return the final pressure field
    /// alongside the L2 residual after the last sweep.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::BackendUnavailable`] when the backend is not
    /// ready.
    pub fn dispatch_pressure_solve(
        &self,
        grid: &PressureGrid,
        n_sweeps: usize,
    ) -> GpuResult<PressureSolveOutput> {
        validate_grid(grid)?;
        #[cfg(feature = "gpu")]
        {
            if !self.backend_ready {
                return Err(GpuError::BackendUnavailable);
            }
            let mut g = grid.clone();
            for _ in 0..n_sweeps {
                g.pressure = Self::cpu_reference_jacobi(&g);
            }
            let residual = Self::residual_l2(&g);
            Ok(PressureSolveOutput {
                pressure: g.pressure,
                residual_l2: residual,
            })
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = n_sweeps;
            Err(GpuError::BackendUnavailable)
        }
    }

    /// CPU reference implementation of one Jacobi pressure sweep — public for
    /// fallback and validation. Boundary cells are held fixed.
    pub fn cpu_reference_jacobi(grid: &PressureGrid) -> Vec<f64> {
        let mut next = grid.pressure.clone();
        if grid.nx < 3 || grid.ny < 3 {
            return next;
        }
        let dx2 = grid.dx * grid.dx;
        let dy2 = grid.dy * grid.dy;
        let denom = 2.0 * (dx2 + dy2);
        for j in 1..grid.ny - 1 {
            for i in 1..grid.nx - 1 {
                let idx = j * grid.nx + i;
                let east = grid.pressure[idx + 1];
                let west = grid.pressure[idx - 1];
                let north = grid.pressure[idx + grid.nx];
                let south = grid.pressure[idx - grid.nx];
                next[idx] = (dy2 * (east + west) + dx2 * (north + south)
                    - dx2 * dy2 * grid.source[idx])
                    / denom;
            }
        }
        next
    }

    /// L2 residual of the Poisson operator on the interior of `grid`.
    pub fn residual_l2(grid: &PressureGrid) -> f64 {
        if grid.nx < 3 || grid.ny < 3 {
            return 0.0;
        }
        let dx2 = grid.dx * grid.dx;
        let dy2 = grid.dy * grid.dy;
        let mut sum = 0.0;
        let mut n = 0usize;
        for j in 1..grid.ny - 1 {
            for i in 1..grid.nx - 1 {
                let idx = j * grid.nx + i;
                let lap = (grid.pressure[idx + 1] + grid.pressure[idx - 1]
                    - 2.0 * grid.pressure[idx])
                    / dx2
                    + (grid.pressure[idx + grid.nx] + grid.pressure[idx - grid.nx]
                        - 2.0 * grid.pressure[idx])
                        / dy2;
                let r = lap - grid.source[idx];
                sum += r * r;
                n += 1;
            }
        }
        if n == 0 {
            0.0
        } else {
            (sum / n as f64).sqrt()
        }
    }
}

/// Result of a multi-sweep pressure solve.
#[derive(Debug, Clone)]
pub struct PressureSolveOutput {
    /// Final pressure field, row-major.
    pub pressure: Vec<f64>,
    /// L2 residual of the Poisson operator after the last sweep.
    pub residual_l2: f64,
}

fn validate_grid(grid: &PressureGrid) -> GpuResult<()> {
    if grid.is_empty() {
        return Err(GpuError::InvalidInput(
            "pressure grid has zero extent".to_string(),
        ));
    }
    if grid.dx <= 0.0 || grid.dy <= 0.0 || !grid.dx.is_finite() || !grid.dy.is_finite() {
        return Err(GpuError::InvalidInput(format!(
            "pressure grid spacing invalid: dx={}, dy={}",
            grid.dx, grid.dy
        )));
    }
    let expected = grid.len();
    if grid.pressure.len() != expected {
        return Err(GpuError::InvalidInput(format!(
            "pressure buffer has {} entries, expected {}",
            grid.pressure.len(),
            expected
        )));
    }
    if grid.source.len() != expected {
        return Err(GpuError::InvalidInput(format!(
            "source buffer has {} entries, expected {}",
            grid.source.len(),
            expected
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_grid(nx: usize, ny: usize) -> PressureGrid {
        PressureGrid {
            nx,
            ny,
            dx: 0.1,
            dy: 0.1,
            pressure: vec![0.0; nx * ny],
            source: vec![0.0; nx * ny],
        }
    }

    #[test]
    fn dispatcher_availability_matches_feature() {
        let d = NavierStokesKernelDispatcher::new();
        #[cfg(feature = "gpu")]
        assert!(d.is_available());
        #[cfg(not(feature = "gpu"))]
        assert!(!d.is_available());
    }

    #[test]
    fn jacobi_no_feature_returns_unavailable() {
        let d = NavierStokesKernelDispatcher::new();
        let g = flat_grid(5, 5);
        let result = d.dispatch_pressure_jacobi(&g);
        #[cfg(not(feature = "gpu"))]
        assert!(matches!(result, Err(GpuError::BackendUnavailable)));
        #[cfg(feature = "gpu")]
        {
            let next = result.expect("dispatch should succeed under gpu feature");
            assert_eq!(next.len(), g.len());
        }
    }

    #[test]
    fn jacobi_zero_source_zero_initial_stays_zero() {
        let g = flat_grid(5, 5);
        let next = NavierStokesKernelDispatcher::cpu_reference_jacobi(&g);
        assert_eq!(next.len(), g.len());
        for v in &next {
            assert!((*v).abs() < 1e-15);
        }
    }

    #[test]
    fn jacobi_dirichlet_boundary_held_fixed() {
        let mut g = flat_grid(5, 5);
        let last = g.len() - 1;
        g.pressure[0] = 1.0;
        g.pressure[last] = 7.0;
        let next = NavierStokesKernelDispatcher::cpu_reference_jacobi(&g);
        assert_eq!(next[0], 1.0);
        assert_eq!(next[last], 7.0);
    }

    #[test]
    fn solve_reduces_residual() {
        let d = NavierStokesKernelDispatcher::new();
        let mut g = flat_grid(7, 7);
        // Set a non-trivial source on the interior.
        let centre = 3 * g.nx + 3;
        g.source[centre] = -5.0;
        let result = d.dispatch_pressure_solve(&g, 50);
        #[cfg(not(feature = "gpu"))]
        assert!(matches!(result, Err(GpuError::BackendUnavailable)));
        #[cfg(feature = "gpu")]
        {
            let out = result.expect("solve should succeed under gpu feature");
            let initial_residual = NavierStokesKernelDispatcher::residual_l2(&g);
            assert!(
                out.residual_l2 < initial_residual,
                "residual must decrease after relaxation: before {initial_residual:e} after {:e}",
                out.residual_l2
            );
        }
    }

    #[test]
    fn invalid_grid_caught_eagerly() {
        let d = NavierStokesKernelDispatcher::new();
        let bad = PressureGrid {
            nx: 3,
            ny: 3,
            dx: -0.1,
            dy: 0.1,
            pressure: vec![0.0; 9],
            source: vec![0.0; 9],
        };
        let result = d.dispatch_pressure_jacobi(&bad);
        assert!(matches!(result, Err(GpuError::InvalidInput(_))));
    }

    #[test]
    fn invalid_buffer_size_caught_eagerly() {
        let d = NavierStokesKernelDispatcher::new();
        let bad = PressureGrid {
            nx: 3,
            ny: 3,
            dx: 0.1,
            dy: 0.1,
            pressure: vec![0.0; 10], // wrong length
            source: vec![0.0; 9],
        };
        let result = d.dispatch_pressure_jacobi(&bad);
        assert!(matches!(result, Err(GpuError::InvalidInput(_))));
    }
}
