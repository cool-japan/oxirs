//! Integration tests for GPU kernel dispatchers.
//!
//! These tests cover the public dispatcher API. They run identically with
//! and without the `gpu` feature: without it, dispatch methods must return
//! [`GpuError::BackendUnavailable`]; with it, they must produce the same
//! deterministic output as the CPU reference implementations.

use oxirs_physics::gpu::heat_kernel::{HeatGrid, HeatKernelDispatcher};
use oxirs_physics::gpu::navier_stokes_kernel::{NavierStokesKernelDispatcher, PressureGrid};
use oxirs_physics::gpu::stress_assembly::{
    FemElementKind, GpuElementDescriptor, StressAssemblyDispatcher,
};
use oxirs_physics::gpu::{backend_available, GpuError};

#[test]
fn backend_status_is_consistent_across_dispatchers() {
    let s = StressAssemblyDispatcher::new();
    let n = NavierStokesKernelDispatcher::new();
    let h = HeatKernelDispatcher::new();
    let expected = backend_available();
    assert_eq!(s.is_available(), expected);
    assert_eq!(n.is_available(), expected);
    assert_eq!(h.is_available(), expected);
}

#[test]
fn stress_dispatch_matches_feature() {
    let dispatcher = StressAssemblyDispatcher::new();
    let elements = vec![
        GpuElementDescriptor {
            kind: FemElementKind::Bar1D,
            stiffness_scale: 1.0e6,
            mass_scale: 1.0,
        },
        GpuElementDescriptor {
            kind: FemElementKind::Quad2D,
            stiffness_scale: 5.0e5,
            mass_scale: 0.5,
        },
    ];
    let result = dispatcher.dispatch_stiffness_assembly(&elements);
    if backend_available() {
        let contribs = result.expect("dispatch should succeed under gpu feature");
        assert_eq!(contribs.len(), elements.len());
        // Bar1D has 2 DoFs.
        assert_eq!(contribs[0].dofs, 2);
        // Quad2D has 8 DoFs.
        assert_eq!(contribs[1].dofs, 8);
    } else {
        assert!(matches!(result, Err(GpuError::BackendUnavailable)));
    }
}

#[test]
fn pressure_solve_drives_residual_down_when_backend_available() {
    let dispatcher = NavierStokesKernelDispatcher::new();
    let nx = 11usize;
    let ny = 11usize;
    let mut grid = PressureGrid {
        nx,
        ny,
        dx: 0.1,
        dy: 0.1,
        pressure: vec![0.0; nx * ny],
        source: vec![0.0; nx * ny],
    };
    grid.source[5 * nx + 5] = -1.0; // central forcing

    let result = dispatcher.dispatch_pressure_solve(&grid, 200);
    if backend_available() {
        let out = result.expect("solve should succeed under gpu feature");
        assert_eq!(out.pressure.len(), grid.pressure.len());
        // Solver should have driven the L2 residual well below the
        // forcing magnitude.
        assert!(out.residual_l2.is_finite());
        assert!(out.residual_l2 < 1.0e-2, "residual = {:e}", out.residual_l2);
    } else {
        assert!(matches!(result, Err(GpuError::BackendUnavailable)));
    }
}

#[test]
fn cpu_reference_pressure_jacobi_is_callable_without_feature() {
    let nx = 5usize;
    let ny = 5usize;
    let grid = PressureGrid {
        nx,
        ny,
        dx: 0.1,
        dy: 0.1,
        pressure: vec![0.0; nx * ny],
        source: vec![0.0; nx * ny],
    };
    // The reference implementation is a public function — usable as a
    // pure-Rust CPU fallback even when the gpu feature is disabled.
    let next = NavierStokesKernelDispatcher::cpu_reference_jacobi(&grid);
    assert_eq!(next.len(), grid.pressure.len());
    for v in next {
        assert_eq!(v, 0.0);
    }
}

#[test]
fn heat_dispatch_matches_feature() {
    let dispatcher = HeatKernelDispatcher::new();
    let nx = 7usize;
    let ny = 7usize;
    let mut grid = HeatGrid {
        nx,
        ny,
        dx: 0.1,
        dy: 0.1,
        dt: 1.0e-3,
        alpha: 1.0e-4,
        temperature: vec![0.0; nx * ny],
    };
    let centre = 3 * nx + 3;
    grid.temperature[centre] = 1.0;

    let result = dispatcher.dispatch_step(&grid);
    if backend_available() {
        let next = result.expect("dispatch should succeed under gpu feature");
        assert_eq!(next.len(), grid.temperature.len());
        // Hot spot must cool, neighbours must warm.
        assert!(next[centre] < 1.0);
        assert!(next[centre - 1] > 0.0);
    } else {
        assert!(matches!(result, Err(GpuError::BackendUnavailable)));
    }
}

#[test]
fn heat_cpu_reference_is_always_callable() {
    let nx = 5usize;
    let ny = 5usize;
    let mut grid = HeatGrid {
        nx,
        ny,
        dx: 0.1,
        dy: 0.1,
        dt: 1.0e-3,
        alpha: 1.0e-4,
        temperature: vec![0.0; nx * ny],
    };
    // Set a uniform value; the CPU reference must preserve it (no flux).
    grid.temperature.fill(273.15);
    let next = HeatKernelDispatcher::cpu_reference_step(&grid);
    for v in next {
        assert!((v - 273.15).abs() < 1e-12);
    }
}

#[test]
fn cpu_reference_heat_step_diffuses_centre_into_neighbours() {
    let nx = 5usize;
    let ny = 5usize;
    let mut grid = HeatGrid {
        nx,
        ny,
        dx: 0.1,
        dy: 0.1,
        dt: 1.0e-3,
        alpha: 1.0e-4,
        temperature: vec![0.0; nx * ny],
    };
    let centre = 2 * nx + 2;
    grid.temperature[centre] = 10.0;
    let next = HeatKernelDispatcher::cpu_reference_step(&grid);
    assert!(next[centre] < 10.0);
    assert!(next[centre - 1] > 0.0);
}

#[test]
fn invalid_input_caught_eagerly_regardless_of_feature() {
    // Negative spacing must always trip InvalidInput before the feature
    // gate is consulted.
    let dispatcher = NavierStokesKernelDispatcher::new();
    let bad = PressureGrid {
        nx: 3,
        ny: 3,
        dx: -1.0,
        dy: 0.1,
        pressure: vec![0.0; 9],
        source: vec![0.0; 9],
    };
    let result = dispatcher.dispatch_pressure_jacobi(&bad);
    assert!(matches!(result, Err(GpuError::InvalidInput(_))));
}
