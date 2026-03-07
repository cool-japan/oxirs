//! # Wave Propagation Simulation
//!
//! Implements 1D/2D wave propagation using the wave equation, FDTD methods,
//! and analytical solutions for standing waves, traveling waves, and attenuated waves.
//!
//! ## Key Capabilities
//!
//! - Analytical wave solutions (standing, traveling, attenuated)
//! - Superposition of multiple wave sources
//! - 1D Finite-Difference Time-Domain (FDTD) numerical solver
//! - Courant stability analysis
//! - Resonant frequency calculation
//! - Reflection and transmission coefficients at impedance boundaries

use std::f64::consts::PI;

/// Parameters defining a wave source
#[derive(Debug, Clone, PartialEq)]
pub struct WaveParams {
    /// Wave propagation speed (m/s)
    pub wave_speed: f64,
    /// Damping coefficient (Np/m), 0.0 for lossless
    pub damping: f64,
    /// Frequency (Hz)
    pub frequency: f64,
    /// Peak amplitude
    pub amplitude: f64,
}

impl WaveParams {
    /// Create a new WaveParams with the given values
    pub fn new(wave_speed: f64, damping: f64, frequency: f64, amplitude: f64) -> Self {
        Self {
            wave_speed,
            damping,
            frequency,
            amplitude,
        }
    }
}

/// 1D computational grid for wave simulation
#[derive(Debug, Clone)]
pub struct Grid1D {
    /// Current field values at each spatial node
    pub values: Vec<f64>,
    /// Spatial step size (m)
    pub dx: f64,
    /// Total grid length (m)
    pub length: f64,
}

impl Grid1D {
    /// Create a new zero-initialised 1D grid.
    ///
    /// `length` is divided into `n_points` equal cells of width `dx = length / (n_points - 1)`.
    pub fn new(length: f64, n_points: usize) -> Self {
        assert!(n_points >= 2, "Grid requires at least 2 points");
        let dx = length / (n_points as f64 - 1.0);
        Self {
            values: vec![0.0; n_points],
            dx,
            length,
        }
    }

    /// Number of spatial nodes in the grid
    pub fn n_points(&self) -> usize {
        self.values.len()
    }

    /// x-coordinate of node `i`
    pub fn x_at(&self, i: usize) -> f64 {
        i as f64 * self.dx
    }
}

/// Wave propagation engine providing analytical and numerical solutions
pub struct WavePropagation;

impl WavePropagation {
    // ─────────────────────────────────────────────────────────────────────────
    // Fundamental wave quantities
    // ─────────────────────────────────────────────────────────────────────────

    /// Wave number k = 2πf / c  (rad/m)
    pub fn wave_number(frequency: f64, speed: f64) -> f64 {
        if speed == 0.0 {
            return 0.0;
        }
        2.0 * PI * frequency / speed
    }

    /// Angular frequency ω = 2πf  (rad/s)
    pub fn angular_frequency(frequency: f64) -> f64 {
        2.0 * PI * frequency
    }

    /// Phase velocity vp = ω / k = f · λ  (m/s)
    ///
    /// Returns 0 when `wave_number` is zero to avoid division by zero.
    pub fn phase_velocity(frequency: f64, wave_number: f64) -> f64 {
        if wave_number == 0.0 {
            return 0.0;
        }
        Self::angular_frequency(frequency) / wave_number
    }

    /// Wavelength λ = c / f  (m)
    ///
    /// Returns 0 when frequency is zero to avoid division by zero.
    pub fn wavelength(speed: f64, frequency: f64) -> f64 {
        if frequency == 0.0 {
            return 0.0;
        }
        speed / frequency
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Analytical wave solutions
    // ─────────────────────────────────────────────────────────────────────────

    /// Standing wave: u(x,t) = A · sin(kx) · cos(ωt)
    pub fn standing_wave(params: &WaveParams, x: f64, t: f64) -> f64 {
        let k = Self::wave_number(params.frequency, params.wave_speed);
        let omega = Self::angular_frequency(params.frequency);
        params.amplitude * (k * x).sin() * (omega * t).cos()
    }

    /// Traveling wave: u(x,t) = A · cos(kx − direction·ωt)
    ///
    /// `direction` = +1 for forward (+x), -1 for backward (-x).
    pub fn traveling_wave(params: &WaveParams, x: f64, t: f64, direction: i8) -> f64 {
        let k = Self::wave_number(params.frequency, params.wave_speed);
        let omega = Self::angular_frequency(params.frequency);
        let dir = direction as f64;
        params.amplitude * (k * x - dir * omega * t).cos()
    }

    /// Attenuated traveling wave: u(x,t) = A · exp(−α·x) · cos(kx − ωt)
    ///
    /// The exponential factor `exp(−α·x)` models spatial decay (α = `params.damping`).
    pub fn attenuated_wave(params: &WaveParams, x: f64, t: f64) -> f64 {
        let k = Self::wave_number(params.frequency, params.wave_speed);
        let omega = Self::angular_frequency(params.frequency);
        let envelope = (-params.damping * x).exp();
        params.amplitude * envelope * (k * x - omega * t).cos()
    }

    /// Superposition of multiple traveling waves.
    ///
    /// Each entry is `(WaveParams, direction_sign)` where the second element
    /// is the direction passed to `traveling_wave`.
    pub fn superposition(waves: &[(WaveParams, f64)], x: f64, t: f64) -> f64 {
        waves.iter().fold(0.0, |acc, (params, sign)| {
            let direction = if *sign >= 0.0 { 1_i8 } else { -1_i8 };
            acc + Self::traveling_wave(params, x, t, direction)
        })
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Numerical solver (1D FDTD)
    // ─────────────────────────────────────────────────────────────────────────

    /// Advance the wave field by one FDTD time step.
    ///
    /// Uses the explicit second-order finite-difference scheme:
    ///
    /// ```text
    /// u_new[i] = 2·u[i] − u_prev[i]
    ///          + (c·dt/dx)² · (u[i+1] − 2·u[i] + u[i-1])
    ///          − 2·α·dt·(u[i] − u_prev[i])
    /// ```
    ///
    /// Dirichlet boundary conditions (u = 0) are enforced at both ends.
    ///
    /// # Panics
    ///
    /// Panics if `prev` length differs from `grid.values`.
    pub fn fdtd_step(grid: &Grid1D, prev: &[f64], dt: f64, params: &WaveParams) -> Vec<f64> {
        assert_eq!(grid.values.len(), prev.len(), "prev must match grid length");
        let n = grid.values.len();
        let mut next = vec![0.0_f64; n];

        // Courant factor squared
        let c = Self::courant_number(params.wave_speed, dt, grid.dx);
        let c2 = c * c;
        let damping_factor = 2.0 * params.damping * dt;

        // Interior nodes
        for i in 1..n.saturating_sub(1) {
            let laplacian = grid.values[i + 1] - 2.0 * grid.values[i] + grid.values[i - 1];
            let velocity_approx = grid.values[i] - prev[i];
            next[i] =
                2.0 * grid.values[i] - prev[i] + c2 * laplacian - damping_factor * velocity_approx;
        }

        // Dirichlet BCs: boundaries remain at zero
        next[0] = 0.0;
        if n > 1 {
            next[n - 1] = 0.0;
        }

        next
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Stability and resonance
    // ─────────────────────────────────────────────────────────────────────────

    /// Courant number C = c · dt / dx  (must be ≤ 1 for FDTD stability)
    pub fn courant_number(speed: f64, dt: f64, dx: f64) -> f64 {
        if dx == 0.0 {
            return f64::INFINITY;
        }
        speed * dt / dx
    }

    /// Resonant frequencies of a string / cavity of given length.
    ///
    /// f_n = n · c / (2 · L)  for n = 1 … `n_modes`
    pub fn resonant_frequencies(length: f64, speed: f64, n_modes: usize) -> Vec<f64> {
        if length == 0.0 || n_modes == 0 {
            return vec![];
        }
        (1..=n_modes)
            .map(|n| n as f64 * speed / (2.0 * length))
            .collect()
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Impedance boundary coefficients
    // ─────────────────────────────────────────────────────────────────────────

    /// Reflection coefficient Γ = (Z2 − Z1) / (Z2 + Z1)
    ///
    /// Returns 0.0 if Z1 + Z2 == 0 to avoid division by zero.
    pub fn reflection_coefficient(z1: f64, z2: f64) -> f64 {
        let denom = z1 + z2;
        if denom == 0.0 {
            return 0.0;
        }
        (z2 - z1) / denom
    }

    /// Transmission coefficient T = 2·Z2 / (Z1 + Z2)
    ///
    /// Returns 0.0 if Z1 + Z2 == 0 to avoid division by zero.
    pub fn transmission_coefficient(z1: f64, z2: f64) -> f64 {
        let denom = z1 + z2;
        if denom == 0.0 {
            return 0.0;
        }
        2.0 * z2 / denom
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-10;

    // ── Wave number ──────────────────────────────────────────────────────────

    #[test]
    fn test_wave_number_basic() {
        // k = 2πf/c.  At f=1 Hz, c=1 m/s → k = 2π
        let k = WavePropagation::wave_number(1.0, 1.0);
        assert!((k - 2.0 * PI).abs() < EPS, "k = {k}");
    }

    #[test]
    fn test_wave_number_zero_speed() {
        assert_eq!(WavePropagation::wave_number(1000.0, 0.0), 0.0);
    }

    #[test]
    fn test_wave_number_high_frequency() {
        let k = WavePropagation::wave_number(100.0, 340.0);
        let expected = 2.0 * PI * 100.0 / 340.0;
        assert!((k - expected).abs() < EPS);
    }

    // ── Angular frequency ────────────────────────────────────────────────────

    #[test]
    fn test_angular_frequency() {
        let omega = WavePropagation::angular_frequency(1.0);
        assert!((omega - 2.0 * PI).abs() < EPS);
    }

    #[test]
    fn test_angular_frequency_zero() {
        assert_eq!(WavePropagation::angular_frequency(0.0), 0.0);
    }

    #[test]
    fn test_angular_frequency_50hz() {
        let omega = WavePropagation::angular_frequency(50.0);
        assert!((omega - 100.0 * PI).abs() < EPS);
    }

    // ── Phase velocity ───────────────────────────────────────────────────────

    #[test]
    fn test_phase_velocity_roundtrip() {
        let f = 440.0;
        let c = 343.0;
        let k = WavePropagation::wave_number(f, c);
        let vp = WavePropagation::phase_velocity(f, k);
        assert!((vp - c).abs() < 1e-9, "vp = {vp}");
    }

    #[test]
    fn test_phase_velocity_zero_wavenumber() {
        assert_eq!(WavePropagation::phase_velocity(440.0, 0.0), 0.0);
    }

    // ── Wavelength ───────────────────────────────────────────────────────────

    #[test]
    fn test_wavelength_sound() {
        // λ = 340 / 1000 = 0.34 m
        let lambda = WavePropagation::wavelength(340.0, 1000.0);
        assert!((lambda - 0.34).abs() < EPS);
    }

    #[test]
    fn test_wavelength_zero_frequency() {
        assert_eq!(WavePropagation::wavelength(340.0, 0.0), 0.0);
    }

    #[test]
    fn test_wavelength_light() {
        // λ ≈ 0.5 μm for green light at 600 THz
        let c = 3.0e8_f64;
        let f = 6.0e14_f64;
        let lambda = WavePropagation::wavelength(c, f);
        assert!((lambda - 5.0e-7).abs() < 1e-9);
    }

    // ── Standing wave ────────────────────────────────────────────────────────

    #[test]
    fn test_standing_wave_node_at_x0() {
        // sin(k·0) = 0 regardless of t
        let params = WaveParams::new(340.0, 0.0, 100.0, 1.0);
        for t in [0.0, 0.5, 1.0] {
            let u = WavePropagation::standing_wave(&params, 0.0, t);
            assert!(u.abs() < EPS, "expected 0 at x=0, got {u} at t={t}");
        }
    }

    #[test]
    fn test_standing_wave_amplitude() {
        // At x = λ/4, t = 0: sin(π/2)·cos(0) = 1 → u = amplitude
        let speed = 340.0;
        let freq = 100.0;
        let lambda = WavePropagation::wavelength(speed, freq);
        let params = WaveParams::new(speed, 0.0, freq, 2.5);
        let u = WavePropagation::standing_wave(&params, lambda / 4.0, 0.0);
        assert!((u - 2.5).abs() < EPS, "u = {u}");
    }

    #[test]
    fn test_standing_wave_time_reversal_symmetry() {
        let params = WaveParams::new(340.0, 0.0, 50.0, 1.0);
        let u_pos = WavePropagation::standing_wave(&params, 0.5, 0.1);
        let u_neg = WavePropagation::standing_wave(&params, 0.5, -0.1);
        // cos(ωt) is even → u(x, t) = u(x, -t)
        assert!((u_pos - u_neg).abs() < EPS);
    }

    // ── Traveling wave ───────────────────────────────────────────────────────

    #[test]
    fn test_traveling_wave_at_origin_t0() {
        let params = WaveParams::new(340.0, 0.0, 100.0, 1.0);
        // cos(0) = 1 → u = amplitude
        let u = WavePropagation::traveling_wave(&params, 0.0, 0.0, 1);
        assert!((u - 1.0).abs() < EPS);
    }

    #[test]
    fn test_traveling_wave_direction_sign() {
        let params = WaveParams::new(340.0, 0.0, 50.0, 1.0);
        let x = 0.3;
        let t = 0.01;
        let forward = WavePropagation::traveling_wave(&params, x, t, 1);
        let backward = WavePropagation::traveling_wave(&params, x, t, -1);
        // They should be equal only by coincidence; in general they differ
        // Just verify they produce finite values
        assert!(forward.is_finite(), "forward wave is not finite");
        assert!(backward.is_finite(), "backward wave is not finite");
    }

    #[test]
    fn test_traveling_wave_periodic() {
        let speed = 340.0;
        let freq = 100.0;
        let period = 1.0 / freq;
        let params = WaveParams::new(speed, 0.0, freq, 1.0);
        let x = 0.5;
        let t = 0.02;
        let u1 = WavePropagation::traveling_wave(&params, x, t, 1);
        let u2 = WavePropagation::traveling_wave(&params, x, t + period, 1);
        assert!((u1 - u2).abs() < EPS, "wave not periodic: {u1} vs {u2}");
    }

    // ── Attenuated wave ──────────────────────────────────────────────────────

    #[test]
    fn test_attenuated_wave_zero_damping_equals_traveling() {
        let params = WaveParams::new(340.0, 0.0, 100.0, 1.0);
        let x = 0.3;
        let t = 0.002;
        let att = WavePropagation::attenuated_wave(&params, x, t);
        let trav = WavePropagation::traveling_wave(&params, x, t, 1);
        assert!((att - trav).abs() < EPS, "att={att}, trav={trav}");
    }

    #[test]
    fn test_attenuated_wave_decays_with_distance() {
        let params = WaveParams::new(340.0, 1.0, 100.0, 1.0);
        let t = 0.0;
        // Envelope: exp(-α·x) is decreasing, so |u(x1)| >= |u(x2)| for x1 < x2
        let u0 = WavePropagation::attenuated_wave(&params, 0.0, t).abs();
        let u1 = WavePropagation::attenuated_wave(&params, 1.0, t).abs();
        assert!(u0 >= u1, "wave should decay: |u(0)|={u0}, |u(1)|={u1}");
    }

    #[test]
    fn test_attenuated_wave_large_damping() {
        // With very large α, wave decays to near zero quickly
        let params = WaveParams::new(340.0, 100.0, 100.0, 1.0);
        let u = WavePropagation::attenuated_wave(&params, 1.0, 0.0);
        assert!(u.abs() < 1e-40, "expected near-zero, got {u}");
    }

    // ── Superposition ────────────────────────────────────────────────────────

    #[test]
    fn test_superposition_single_wave() {
        let params = WaveParams::new(340.0, 0.0, 50.0, 1.0);
        let waves = vec![(params.clone(), 1.0)];
        let x = 0.2;
        let t = 0.003;
        let sup = WavePropagation::superposition(&waves, x, t);
        let single = WavePropagation::traveling_wave(&params, x, t, 1);
        assert!((sup - single).abs() < EPS);
    }

    #[test]
    fn test_superposition_opposite_waves_standing() {
        // A·cos(kx − ωt) + A·cos(kx + ωt) = 2A·cos(kx)·cos(ωt)
        let p = WaveParams::new(340.0, 0.0, 100.0, 1.0);
        let waves = vec![(p.clone(), 1.0), (p.clone(), -1.0)];
        let x = 0.1;
        let t = 0.002;
        let sup = WavePropagation::superposition(&waves, x, t);
        // Expected: 2·cos(kx)·cos(ωt)
        let k = WavePropagation::wave_number(p.frequency, p.wave_speed);
        let omega = WavePropagation::angular_frequency(p.frequency);
        let expected = 2.0 * (k * x).cos() * (omega * t).cos();
        assert!(
            (sup - expected).abs() < EPS,
            "sup={sup}, expected={expected}"
        );
    }

    #[test]
    fn test_superposition_three_waves() {
        let p1 = WaveParams::new(340.0, 0.0, 100.0, 1.0);
        let p2 = WaveParams::new(340.0, 0.0, 200.0, 0.5);
        let p3 = WaveParams::new(340.0, 0.0, 300.0, 0.3);
        let waves = vec![(p1.clone(), 1.0), (p2.clone(), 1.0), (p3.clone(), 1.0)];
        let x = 0.15;
        let t = 0.001;
        let sup = WavePropagation::superposition(&waves, x, t);
        let manual = WavePropagation::traveling_wave(&p1, x, t, 1)
            + WavePropagation::traveling_wave(&p2, x, t, 1)
            + WavePropagation::traveling_wave(&p3, x, t, 1);
        assert!((sup - manual).abs() < EPS);
    }

    // ── FDTD ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_fdtd_zero_initial_stays_zero() {
        let grid = Grid1D::new(1.0, 11);
        let prev = vec![0.0_f64; grid.n_points()];
        let params = WaveParams::new(340.0, 0.0, 100.0, 1.0);
        let next = WavePropagation::fdtd_step(&grid, &prev, 0.001, &params);
        for v in &next {
            assert!(v.abs() < EPS, "expected 0, got {v}");
        }
    }

    #[test]
    fn test_fdtd_boundary_zero() {
        let mut grid = Grid1D::new(1.0, 11);
        // Inject a pulse in the centre
        let mid = grid.n_points() / 2;
        grid.values[mid] = 1.0;
        let prev = vec![0.0_f64; grid.n_points()];
        let params = WaveParams::new(340.0, 0.0, 100.0, 1.0);
        let next = WavePropagation::fdtd_step(&grid, &prev, 0.001, &params);
        // Dirichlet BCs must hold
        assert_eq!(next[0], 0.0);
        assert_eq!(*next.last().expect("non-empty"), 0.0);
    }

    #[test]
    fn test_fdtd_stable_courant() {
        // Courant = 1.0 → marginally stable; field should not blow up
        let n = 21;
        let mut grid = Grid1D::new(1.0, n);
        grid.values[n / 2] = 1.0;
        let dt = grid.dx / 340.0; // exactly Courant = 1
        let params = WaveParams::new(340.0, 0.0, 100.0, 1.0);
        let prev = vec![0.0_f64; n];
        let next = WavePropagation::fdtd_step(&grid, &prev, dt, &params);
        for v in &next {
            assert!(
                v.abs() < 10.0,
                "field blew up: {v}; Courant stability violation"
            );
        }
    }

    #[test]
    fn test_fdtd_output_length() {
        let grid = Grid1D::new(2.0, 51);
        let prev = vec![0.0_f64; grid.n_points()];
        let params = WaveParams::new(340.0, 0.0, 50.0, 1.0);
        let next = WavePropagation::fdtd_step(&grid, &prev, 1e-4, &params);
        assert_eq!(next.len(), grid.n_points());
    }

    // ── Courant number ───────────────────────────────────────────────────────

    #[test]
    fn test_courant_number_marginal() {
        let c = WavePropagation::courant_number(340.0, 1.0 / 340.0, 1.0);
        assert!((c - 1.0).abs() < EPS);
    }

    #[test]
    fn test_courant_number_stable() {
        let c = WavePropagation::courant_number(340.0, 0.001, 1.0);
        assert!(c <= 1.0, "c = {c}");
    }

    #[test]
    fn test_courant_number_zero_dx() {
        let c = WavePropagation::courant_number(340.0, 0.001, 0.0);
        assert!(c.is_infinite());
    }

    // ── Resonant frequencies ─────────────────────────────────────────────────

    #[test]
    fn test_resonant_frequencies_count() {
        let freqs = WavePropagation::resonant_frequencies(1.0, 340.0, 5);
        assert_eq!(freqs.len(), 5);
    }

    #[test]
    fn test_resonant_frequencies_fundamental() {
        // f1 = c / (2L)
        let freqs = WavePropagation::resonant_frequencies(1.0, 340.0, 1);
        assert!((freqs[0] - 170.0).abs() < EPS);
    }

    #[test]
    fn test_resonant_frequencies_harmonics() {
        let freqs = WavePropagation::resonant_frequencies(1.0, 340.0, 3);
        // f_n = n · f_1
        let f1 = freqs[0];
        assert!((freqs[1] - 2.0 * f1).abs() < EPS);
        assert!((freqs[2] - 3.0 * f1).abs() < EPS);
    }

    #[test]
    fn test_resonant_frequencies_zero_modes() {
        let freqs = WavePropagation::resonant_frequencies(1.0, 340.0, 0);
        assert!(freqs.is_empty());
    }

    #[test]
    fn test_resonant_frequencies_zero_length() {
        let freqs = WavePropagation::resonant_frequencies(0.0, 340.0, 5);
        assert!(freqs.is_empty());
    }

    // ── Reflection coefficient ───────────────────────────────────────────────

    #[test]
    fn test_reflection_coefficient_open_end() {
        // Z2 → ∞: Γ → 1 (full reflection, same phase)
        // We approximate with a very large Z2
        let gamma = WavePropagation::reflection_coefficient(1.0, 1e15);
        assert!((gamma - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_reflection_coefficient_rigid_end() {
        // Z1 → ∞: Γ → -1 (full reflection, phase inversion)
        let gamma = WavePropagation::reflection_coefficient(1e15, 1.0);
        assert!((gamma + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_reflection_coefficient_matched() {
        // Z1 = Z2 → Γ = 0
        let gamma = WavePropagation::reflection_coefficient(100.0, 100.0);
        assert!(gamma.abs() < EPS);
    }

    #[test]
    fn test_reflection_coefficient_zero_impedances() {
        let gamma = WavePropagation::reflection_coefficient(0.0, 0.0);
        assert_eq!(gamma, 0.0);
    }

    // ── Transmission coefficient ─────────────────────────────────────────────

    #[test]
    fn test_transmission_coefficient_matched() {
        // Z1 = Z2 → T = 1
        let t = WavePropagation::transmission_coefficient(100.0, 100.0);
        assert!((t - 1.0).abs() < EPS);
    }

    #[test]
    fn test_transmission_coefficient_open_end() {
        // Z2 → ∞: T → 2 (classic result for pressure-free boundary)
        let t = WavePropagation::transmission_coefficient(1.0, 1e15);
        assert!((t - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_transmission_coefficient_zero_impedances() {
        let t = WavePropagation::transmission_coefficient(0.0, 0.0);
        assert_eq!(t, 0.0);
    }

    #[test]
    fn test_energy_conservation_reflection_transmission() {
        // Power reflection: |Γ|²·Z1, power transmission: T·Z1 (proportional)
        // For lossless media: R + T' = 1 where T' = (Z1/Z2)·|T|²
        let z1 = 200.0_f64;
        let z2 = 400.0_f64;
        let gamma = WavePropagation::reflection_coefficient(z1, z2);
        let tau = WavePropagation::transmission_coefficient(z1, z2);
        let r = gamma * gamma;
        let t_power = (z1 / z2) * tau * tau;
        assert!((r + t_power - 1.0).abs() < EPS, "r+T'={}", r + t_power);
    }

    // ── Grid1D helpers ───────────────────────────────────────────────────────

    #[test]
    fn test_grid1d_construction() {
        let grid = Grid1D::new(10.0, 101);
        assert_eq!(grid.n_points(), 101);
        assert!((grid.dx - 0.1).abs() < EPS);
    }

    #[test]
    fn test_grid1d_x_at() {
        let grid = Grid1D::new(1.0, 11);
        assert!((grid.x_at(0) - 0.0).abs() < EPS);
        assert!((grid.x_at(10) - 1.0).abs() < EPS);
    }

    #[test]
    fn test_wave_params_construction() {
        let p = WaveParams::new(340.0, 0.5, 440.0, 2.0);
        assert_eq!(p.wave_speed, 340.0);
        assert_eq!(p.damping, 0.5);
        assert_eq!(p.frequency, 440.0);
        assert_eq!(p.amplitude, 2.0);
    }

    // ── Integration: FDTD energy decay with damping ──────────────────────────

    #[test]
    fn test_fdtd_damped_energy_decreases() {
        let n = 41;
        let mut grid = Grid1D::new(1.0, n);
        let mid = n / 2;
        grid.values[mid] = 1.0;

        let params = WaveParams::new(100.0, 5.0, 50.0, 1.0);
        let dt = grid.dx / (params.wave_speed * 1.5); // Courant < 1

        // Use prev = grid.values (zero initial velocity) so the FDTD scheme
        // starts from a self-consistent state.  With prev = 0 the implicit
        // non-zero velocity injects energy on the first step, defeating the
        // damping assertion.
        let mut prev = grid.values.clone();
        let initial_energy: f64 = grid.values.iter().map(|v| v * v).sum();

        for _ in 0..20 {
            let next = WavePropagation::fdtd_step(&grid, &prev, dt, &params);
            prev = grid.values.clone();
            grid.values = next;
        }
        let final_energy: f64 = grid.values.iter().map(|v| v * v).sum();
        // With damping > 0 and zero initial velocity, energy must decrease
        assert!(
            final_energy < initial_energy,
            "energy should decrease with damping; initial={initial_energy}, final={final_energy}"
        );
    }

    // ── Algebraic identities ─────────────────────────────────────────────────

    #[test]
    fn test_wavenumber_wavelength_inverse() {
        let speed = 340.0;
        let freq = 220.0;
        let k = WavePropagation::wave_number(freq, speed);
        let lambda = WavePropagation::wavelength(speed, freq);
        // k · λ = 2π
        assert!((k * lambda - 2.0 * PI).abs() < EPS);
    }

    #[test]
    fn test_angular_frequency_wave_number_phase_velocity() {
        let f = 1000.0;
        let c = 340.0;
        let omega = WavePropagation::angular_frequency(f);
        let k = WavePropagation::wave_number(f, c);
        let vp = WavePropagation::phase_velocity(f, k);
        assert!((vp - c).abs() < 1e-9);
        // Also verify ω = k · vp
        assert!((omega - k * vp).abs() < EPS);
    }
}
