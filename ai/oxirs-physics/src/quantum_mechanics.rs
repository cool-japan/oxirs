//! Basic quantum mechanics calculations.
//!
//! Implements:
//! - Particle-in-a-box energy levels (1D, 2D, 3D)
//! - Quantum harmonic oscillator (energy eigenvalues, wavefunctions)
//! - Tunneling probability (rectangular barrier, WKB approximation)
//! - Expectation values (`<x>`, `<p>`, `<x²>`, `<p²>`)
//! - Heisenberg uncertainty principle verification
//! - Hydrogen atom energy levels
//! - Spin-1/2 algebra (Pauli matrices, spin states)
//! - Density matrix operations (pure/mixed state detection)

use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Physical constants
// ---------------------------------------------------------------------------

/// Reduced Planck constant (ℏ) in J·s.
pub const HBAR: f64 = 1.054_571_817e-34;

/// Electron mass in kg.
pub const ELECTRON_MASS: f64 = 9.109_383_701_5e-31;

/// Bohr radius in metres.
pub const BOHR_RADIUS: f64 = 5.291_772_109_03e-11;

/// Rydberg energy in eV.
pub const RYDBERG_EV: f64 = 13.605_693_122_994;

// ---------------------------------------------------------------------------
// Particle in a Box
// ---------------------------------------------------------------------------

/// Energy of a 1D particle-in-a-box at quantum number n (n ≥ 1).
///
/// E_n = n² π² ℏ² / (2 m L²)
///
/// `mass` in kg, `length` in metres. Returns energy in joules.
pub fn box_1d_energy(n: u64, mass: f64, length: f64) -> f64 {
    let n = n as f64;
    n * n * PI * PI * HBAR * HBAR / (2.0 * mass * length * length)
}

/// Energy of a 2D particle-in-a-box at quantum numbers (nx, ny).
///
/// E = (nx² + ny²) π² ℏ² / (2 m L²)   (square box with side L).
pub fn box_2d_energy(nx: u64, ny: u64, mass: f64, length: f64) -> f64 {
    let nx2 = (nx as f64) * (nx as f64);
    let ny2 = (ny as f64) * (ny as f64);
    (nx2 + ny2) * PI * PI * HBAR * HBAR / (2.0 * mass * length * length)
}

/// Energy of a 3D particle-in-a-box at quantum numbers (nx, ny, nz).
///
/// E = (nx² + ny² + nz²) π² ℏ² / (2 m L²)   (cubic box with side L).
pub fn box_3d_energy(nx: u64, ny: u64, nz: u64, mass: f64, length: f64) -> f64 {
    let nx2 = (nx as f64) * (nx as f64);
    let ny2 = (ny as f64) * (ny as f64);
    let nz2 = (nz as f64) * (nz as f64);
    (nx2 + ny2 + nz2) * PI * PI * HBAR * HBAR / (2.0 * mass * length * length)
}

/// Wavefunction of 1D particle-in-a-box: ψ_n(x) = sqrt(2/L) sin(nπx/L).
///
/// `x` must be in [0, L].
pub fn box_1d_wavefunction(n: u64, x: f64, length: f64) -> f64 {
    (2.0 / length).sqrt() * (n as f64 * PI * x / length).sin()
}

// ---------------------------------------------------------------------------
// Quantum Harmonic Oscillator
// ---------------------------------------------------------------------------

/// Energy eigenvalue of the quantum harmonic oscillator at level n (n ≥ 0).
///
/// E_n = ℏω(n + 1/2)
///
/// `omega` is the angular frequency in rad/s.
pub fn qho_energy(n: u64, omega: f64) -> f64 {
    HBAR * omega * (n as f64 + 0.5)
}

/// Zero-point energy of the QHO: E_0 = ℏω/2.
pub fn qho_zero_point_energy(omega: f64) -> f64 {
    0.5 * HBAR * omega
}

/// Characteristic length scale of the QHO: a = sqrt(ℏ/(mω)).
pub fn qho_length_scale(mass: f64, omega: f64) -> f64 {
    (HBAR / (mass * omega)).sqrt()
}

/// QHO ground-state wavefunction: ψ_0(x) = (mω/πℏ)^{1/4} exp(−mωx²/(2ℏ)).
pub fn qho_ground_state_wavefunction(x: f64, mass: f64, omega: f64) -> f64 {
    let alpha = mass * omega / HBAR;
    let prefactor = (alpha / PI).sqrt().sqrt(); // (α/π)^{1/4}
    prefactor * (-0.5 * alpha * x * x).exp()
}

// ---------------------------------------------------------------------------
// Tunneling
// ---------------------------------------------------------------------------

/// Tunneling probability through a rectangular barrier (transmission coefficient).
///
/// T = 1 / (1 + V₀² sin²(κa) / (4E(V₀ − E)))   for E < V₀
///
/// where κ = sqrt(2m(V₀ − E)) / ℏ, `barrier_width` = a.
///
/// Returns a value in [0, 1].
pub fn tunneling_probability(
    energy_j: f64,
    barrier_height_j: f64,
    barrier_width: f64,
    mass: f64,
) -> f64 {
    if energy_j >= barrier_height_j {
        // Classically allowed — full transmission in the simple model.
        return 1.0;
    }

    let diff = barrier_height_j - energy_j;
    let kappa = (2.0 * mass * diff).sqrt() / HBAR;
    let ka = kappa * barrier_width;

    // For large ka, use exponential approximation to avoid overflow.
    if ka > 30.0 {
        let approx =
            16.0 * energy_j * diff / (barrier_height_j * barrier_height_j) * (-2.0 * ka).exp();
        return approx.clamp(0.0, 1.0);
    }

    let sinh_val = ka.sinh();
    let denom =
        1.0 + barrier_height_j * barrier_height_j * sinh_val * sinh_val / (4.0 * energy_j * diff);
    (1.0 / denom).clamp(0.0, 1.0)
}

/// WKB approximation for tunneling probability.
///
/// T ≈ exp(−2/ℏ ∫ sqrt(2m(V₀ − E)) dx) = exp(−2κa)
pub fn wkb_tunneling(energy_j: f64, barrier_height_j: f64, barrier_width: f64, mass: f64) -> f64 {
    if energy_j >= barrier_height_j {
        return 1.0;
    }
    let diff = barrier_height_j - energy_j;
    let kappa = (2.0 * mass * diff).sqrt() / HBAR;
    (-2.0 * kappa * barrier_width).exp().clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// Expectation values for QHO
// ---------------------------------------------------------------------------

/// `<x>` for the n-th eigenstate of the QHO = 0 (by symmetry).
pub fn qho_expectation_x(_n: u64) -> f64 {
    0.0
}

/// `<p>` for the n-th eigenstate of the QHO = 0 (by symmetry).
pub fn qho_expectation_p(_n: u64) -> f64 {
    0.0
}

/// <x²> for the n-th eigenstate: <x²> = (ℏ/(2mω))(2n + 1).
pub fn qho_expectation_x_squared(n: u64, mass: f64, omega: f64) -> f64 {
    (HBAR / (2.0 * mass * omega)) * (2.0 * n as f64 + 1.0)
}

/// <p²> for the n-th eigenstate: <p²> = (mωℏ/2)(2n + 1).
pub fn qho_expectation_p_squared(n: u64, mass: f64, omega: f64) -> f64 {
    (mass * omega * HBAR / 2.0) * (2.0 * n as f64 + 1.0)
}

// ---------------------------------------------------------------------------
// Heisenberg uncertainty
// ---------------------------------------------------------------------------

/// Compute Δx · Δp for the n-th QHO eigenstate.
///
/// Should always satisfy Δx·Δp ≥ ℏ/2.
pub fn heisenberg_product(n: u64, mass: f64, omega: f64) -> f64 {
    let x2 = qho_expectation_x_squared(n, mass, omega);
    let p2 = qho_expectation_p_squared(n, mass, omega);
    // Δx = sqrt(<x²> − <x>²) = sqrt(<x²>), similarly for p.
    x2.sqrt() * p2.sqrt()
}

/// Check if the uncertainty product satisfies Heisenberg's inequality.
pub fn heisenberg_satisfied(n: u64, mass: f64, omega: f64) -> bool {
    heisenberg_product(n, mass, omega) >= HBAR / 2.0 - 1e-50 // small epsilon for float
}

// ---------------------------------------------------------------------------
// Hydrogen atom
// ---------------------------------------------------------------------------

/// Hydrogen atom energy level in eV.
///
/// E_n = −13.6 / n² eV
pub fn hydrogen_energy_ev(n: u64) -> f64 {
    if n == 0 {
        return f64::NEG_INFINITY;
    }
    -RYDBERG_EV / (n as f64 * n as f64)
}

/// Photon energy (eV) emitted in a transition from n_upper to n_lower.
///
/// ΔE = 13.6 (1/n_lower² − 1/n_upper²) eV
pub fn hydrogen_transition_energy(n_upper: u64, n_lower: u64) -> f64 {
    if n_lower == 0 || n_upper == 0 {
        return 0.0;
    }
    RYDBERG_EV * (1.0 / (n_lower as f64).powi(2) - 1.0 / (n_upper as f64).powi(2))
}

/// Orbital degeneracy of the n-th shell: g = n².
pub fn hydrogen_degeneracy(n: u64) -> u64 {
    n * n
}

// ---------------------------------------------------------------------------
// Spin-1/2 algebra
// ---------------------------------------------------------------------------

/// Pauli X matrix as `[[0,1],[1,0]]` stored row-major as `[a11, a12, a21, a22]`.
pub fn pauli_x() -> [f64; 4] {
    [0.0, 1.0, 1.0, 0.0]
}

/// Pauli Y matrix. Real part: `[[0,0],[0,0]]`, Imaginary part: `[[0,-1],[1,0]]`.
/// Stored as (real, imag) pairs: `[(0,0), (0,-1), (0,1), (0,0)]`.
pub fn pauli_y_imag() -> [(f64, f64); 4] {
    [(0.0, 0.0), (0.0, -1.0), (0.0, 1.0), (0.0, 0.0)]
}

/// Pauli Z matrix: `[[1,0],[0,-1]]`.
pub fn pauli_z() -> [f64; 4] {
    [1.0, 0.0, 0.0, -1.0]
}

/// Trace of a 2x2 real matrix stored as `[a11, a12, a21, a22]`.
pub fn trace_2x2(m: &[f64; 4]) -> f64 {
    m[0] + m[3]
}

/// Determinant of a 2x2 real matrix.
pub fn det_2x2(m: &[f64; 4]) -> f64 {
    m[0] * m[3] - m[1] * m[2]
}

/// Multiply two 2x2 real matrices.
pub fn mul_2x2(a: &[f64; 4], b: &[f64; 4]) -> [f64; 4] {
    [
        a[0] * b[0] + a[1] * b[2],
        a[0] * b[1] + a[1] * b[3],
        a[2] * b[0] + a[3] * b[2],
        a[2] * b[1] + a[3] * b[3],
    ]
}

/// Spin-up state |↑⟩ = (1, 0).
pub fn spin_up() -> (f64, f64) {
    (1.0, 0.0)
}

/// Spin-down state |↓⟩ = (0, 1).
pub fn spin_down() -> (f64, f64) {
    (0.0, 1.0)
}

/// Apply Pauli Z to a spin state and return the result.
pub fn apply_pauli_z(state: (f64, f64)) -> (f64, f64) {
    // σ_z |ψ⟩ = [[1,0],[0,-1]] · (a, b) = (a, -b)
    (state.0, -state.1)
}

/// Apply Pauli X to a spin state.
pub fn apply_pauli_x(state: (f64, f64)) -> (f64, f64) {
    // σ_x |ψ⟩ = [[0,1],[1,0]] · (a, b) = (b, a)
    (state.1, state.0)
}

// ---------------------------------------------------------------------------
// Density matrix operations
// ---------------------------------------------------------------------------

/// Construct a pure-state density matrix ρ = |ψ⟩⟨ψ| for a 2-component state.
///
/// Returns the 2x2 matrix as [ρ₁₁, ρ₁₂, ρ₂₁, ρ₂₂].
pub fn density_matrix_pure(state: (f64, f64)) -> [f64; 4] {
    let (a, b) = state;
    [a * a, a * b, b * a, b * b]
}

/// Check if a 2x2 density matrix represents a pure state.
///
/// A state is pure iff Tr(ρ²) = 1.
pub fn is_pure_state(rho: &[f64; 4]) -> bool {
    let rho2 = mul_2x2(rho, rho);
    let tr = trace_2x2(&rho2);
    (tr - 1.0).abs() < 1e-10
}

/// Check if a 2x2 density matrix represents a mixed state.
///
/// A state is mixed iff Tr(ρ²) < 1.
pub fn is_mixed_state(rho: &[f64; 4]) -> bool {
    let rho2 = mul_2x2(rho, rho);
    let tr = trace_2x2(&rho2);
    tr < 1.0 - 1e-10
}

/// Von Neumann entropy: S = −Tr(ρ ln ρ).
///
/// For a 2x2 density matrix, computed from eigenvalues.
pub fn von_neumann_entropy(rho: &[f64; 4]) -> f64 {
    let tr = trace_2x2(rho);
    let det = det_2x2(rho);

    // Eigenvalues of 2x2: λ = (tr ± sqrt(tr² − 4det)) / 2
    let disc = tr * tr - 4.0 * det;
    if disc < 0.0 {
        return 0.0;
    }
    let sqrt_disc = disc.sqrt();
    let lambda1 = (tr + sqrt_disc) / 2.0;
    let lambda2 = (tr - sqrt_disc) / 2.0;

    let mut entropy = 0.0;
    if lambda1 > 1e-15 {
        entropy -= lambda1 * lambda1.ln();
    }
    if lambda2 > 1e-15 {
        entropy -= lambda2 * lambda2.ln();
    }
    entropy
}

/// Purity of a density matrix: Tr(ρ²).
pub fn purity(rho: &[f64; 4]) -> f64 {
    let rho2 = mul_2x2(rho, rho);
    trace_2x2(&rho2)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_approx(actual: f64, expected: f64, tol: f64, msg: &str) {
        let err = (actual - expected).abs();
        assert!(
            err < tol,
            "{msg}: got {actual:.6e}, expected {expected:.6e}, err {err:.6e}"
        );
    }

    // --- Particle in a Box ---

    #[test]
    fn test_box_1d_ground_state() {
        let e1 = box_1d_energy(1, ELECTRON_MASS, 1e-9);
        assert!(e1 > 0.0, "ground state energy must be positive");
    }

    #[test]
    fn test_box_1d_energy_scales_as_n_squared() {
        let e1 = box_1d_energy(1, ELECTRON_MASS, 1e-9);
        let e2 = box_1d_energy(2, ELECTRON_MASS, 1e-9);
        assert_approx(e2 / e1, 4.0, 1e-10, "E2/E1 = 4");
    }

    #[test]
    fn test_box_1d_energy_formula() {
        let l = 1e-9;
        let expected = PI * PI * HBAR * HBAR / (2.0 * ELECTRON_MASS * l * l);
        assert_approx(
            box_1d_energy(1, ELECTRON_MASS, l),
            expected,
            1e-40,
            "1D box formula",
        );
    }

    #[test]
    fn test_box_2d_energy_sum() {
        let l = 1e-9;
        let e11 = box_2d_energy(1, 1, ELECTRON_MASS, l);
        let e1_1d = box_1d_energy(1, ELECTRON_MASS, l);
        assert_approx(e11, 2.0 * e1_1d, 1e-30, "2D (1,1) = 2 * 1D (1)");
    }

    #[test]
    fn test_box_3d_energy_sum() {
        let l = 1e-9;
        let e111 = box_3d_energy(1, 1, 1, ELECTRON_MASS, l);
        let e1_1d = box_1d_energy(1, ELECTRON_MASS, l);
        assert_approx(e111, 3.0 * e1_1d, 1e-30, "3D (1,1,1) = 3 * 1D (1)");
    }

    #[test]
    fn test_box_1d_wavefunction_at_boundary_zero() {
        let psi_0 = box_1d_wavefunction(1, 0.0, 1e-9);
        assert!(psi_0.abs() < 1e-10, "ψ(0) = 0");
    }

    #[test]
    fn test_box_1d_wavefunction_at_boundary_l() {
        let l = 1e-9;
        let psi_l = box_1d_wavefunction(1, l, l);
        assert!(psi_l.abs() < 1e-10, "ψ(L) = 0");
    }

    #[test]
    fn test_box_1d_wavefunction_max_at_center() {
        let l = 1e-9;
        let psi_mid = box_1d_wavefunction(1, l / 2.0, l);
        let psi_quarter = box_1d_wavefunction(1, l / 4.0, l);
        assert!(psi_mid.abs() > psi_quarter.abs(), "ψ peaks at L/2 for n=1");
    }

    // --- Quantum Harmonic Oscillator ---

    #[test]
    fn test_qho_zero_point() {
        let omega = 1e14;
        let e0 = qho_energy(0, omega);
        let zpe = qho_zero_point_energy(omega);
        assert_approx(e0, zpe, 1e-50, "E0 = zero-point energy");
    }

    #[test]
    fn test_qho_energy_spacing() {
        let omega = 1e14;
        let e0 = qho_energy(0, omega);
        let e1 = qho_energy(1, omega);
        assert_approx(e1 - e0, HBAR * omega, 1e-30, "uniform spacing ℏω");
    }

    #[test]
    fn test_qho_energy_half_integer() {
        let omega = 2e14;
        let e3 = qho_energy(3, omega);
        let expected = HBAR * omega * 3.5;
        assert_approx(e3, expected, 1e-50, "E3 = ℏω(3+1/2)");
    }

    #[test]
    fn test_qho_length_scale_positive() {
        let a = qho_length_scale(ELECTRON_MASS, 1e14);
        assert!(a > 0.0, "length scale must be positive");
    }

    #[test]
    fn test_qho_ground_state_normalised() {
        // Numerical check: integrate |ψ|² dx ≈ 1 using simple trapezoid rule.
        let mass = ELECTRON_MASS;
        let omega = 1e14;
        let a = qho_length_scale(mass, omega);
        let n_points = 10000;
        let x_max = 10.0 * a;
        let dx = 2.0 * x_max / n_points as f64;
        let mut integral = 0.0;
        for i in 0..=n_points {
            let x = -x_max + i as f64 * dx;
            let psi = qho_ground_state_wavefunction(x, mass, omega);
            let weight = if i == 0 || i == n_points { 0.5 } else { 1.0 };
            integral += weight * psi * psi * dx;
        }
        assert_approx(integral, 1.0, 0.01, "ψ₀ normalisation");
    }

    // --- Tunneling ---

    #[test]
    fn test_tunneling_above_barrier_is_one() {
        let t = tunneling_probability(2.0, 1.0, 1e-10, ELECTRON_MASS);
        assert_approx(t, 1.0, 1e-10, "E > V₀ → T = 1");
    }

    #[test]
    fn test_tunneling_less_than_one_below_barrier() {
        let t = tunneling_probability(1e-19, 2e-19, 1e-10, ELECTRON_MASS);
        assert!(t > 0.0 && t < 1.0, "0 < T < 1 for E < V₀");
    }

    #[test]
    fn test_tunneling_decreases_with_width() {
        let t1 = tunneling_probability(1e-19, 2e-19, 1e-10, ELECTRON_MASS);
        let t2 = tunneling_probability(1e-19, 2e-19, 2e-10, ELECTRON_MASS);
        assert!(t2 < t1, "wider barrier → lower tunneling");
    }

    #[test]
    fn test_wkb_tunneling_positive() {
        let t = wkb_tunneling(1e-19, 2e-19, 1e-10, ELECTRON_MASS);
        assert!(t > 0.0 && t <= 1.0);
    }

    #[test]
    fn test_wkb_above_barrier() {
        let t = wkb_tunneling(3e-19, 2e-19, 1e-10, ELECTRON_MASS);
        assert_approx(t, 1.0, 1e-10, "WKB: E > V₀ → 1");
    }

    // --- Expectation values ---

    #[test]
    fn test_expectation_x_is_zero() {
        assert_approx(qho_expectation_x(0), 0.0, 1e-15, "<x> = 0");
        assert_approx(qho_expectation_x(5), 0.0, 1e-15, "<x> = 0 for n=5");
    }

    #[test]
    fn test_expectation_p_is_zero() {
        assert_approx(qho_expectation_p(0), 0.0, 1e-15, "<p> = 0");
    }

    #[test]
    fn test_expectation_x2_ground_state() {
        let mass = ELECTRON_MASS;
        let omega = 1e14;
        let x2 = qho_expectation_x_squared(0, mass, omega);
        let expected = HBAR / (2.0 * mass * omega);
        assert_approx(x2, expected, 1e-55, "<x²> ground state");
    }

    #[test]
    fn test_expectation_p2_ground_state() {
        let mass = ELECTRON_MASS;
        let omega = 1e14;
        let p2 = qho_expectation_p_squared(0, mass, omega);
        let expected = mass * omega * HBAR / 2.0;
        assert_approx(p2, expected, 1e-55, "<p²> ground state");
    }

    // --- Heisenberg uncertainty ---

    #[test]
    fn test_heisenberg_ground_state_is_minimum() {
        let mass = ELECTRON_MASS;
        let omega = 1e14;
        let product = heisenberg_product(0, mass, omega);
        // For n=0: Δx·Δp = ℏ/2 exactly.
        assert_approx(product, HBAR / 2.0, 1e-50, "Δx·Δp = ℏ/2 for n=0");
    }

    #[test]
    fn test_heisenberg_satisfied_all_n() {
        let mass = ELECTRON_MASS;
        let omega = 1e14;
        for n in 0..10 {
            assert!(
                heisenberg_satisfied(n, mass, omega),
                "Heisenberg must hold for n={n}"
            );
        }
    }

    #[test]
    fn test_heisenberg_increases_with_n() {
        let mass = ELECTRON_MASS;
        let omega = 1e14;
        let p0 = heisenberg_product(0, mass, omega);
        let p5 = heisenberg_product(5, mass, omega);
        assert!(p5 > p0, "uncertainty grows with n");
    }

    // --- Hydrogen atom ---

    #[test]
    fn test_hydrogen_ground_state_energy() {
        let e1 = hydrogen_energy_ev(1);
        assert_approx(e1, -13.605_693_122_994, 1e-6, "H ground state");
    }

    #[test]
    fn test_hydrogen_energy_scales_as_n_squared() {
        let e1 = hydrogen_energy_ev(1);
        let e2 = hydrogen_energy_ev(2);
        assert_approx(e2 / e1, 0.25, 1e-10, "E2/E1 = 1/4");
    }

    #[test]
    fn test_hydrogen_transition_energy_lyman_alpha() {
        let de = hydrogen_transition_energy(2, 1);
        let expected = RYDBERG_EV * (1.0 - 0.25); // 13.6 * 0.75
        assert_approx(de, expected, 1e-6, "Lyman-α");
    }

    #[test]
    fn test_hydrogen_degeneracy() {
        assert_eq!(hydrogen_degeneracy(1), 1);
        assert_eq!(hydrogen_degeneracy(2), 4);
        assert_eq!(hydrogen_degeneracy(3), 9);
    }

    // --- Spin-1/2 ---

    #[test]
    fn test_pauli_x_squared_is_identity() {
        let sx = pauli_x();
        let sx2 = mul_2x2(&sx, &sx);
        assert_approx(sx2[0], 1.0, 1e-15, "σ_x² = I (1,1)");
        assert_approx(sx2[3], 1.0, 1e-15, "σ_x² = I (2,2)");
        assert_approx(sx2[1], 0.0, 1e-15, "σ_x² = I (1,2)");
        assert_approx(sx2[2], 0.0, 1e-15, "σ_x² = I (2,1)");
    }

    #[test]
    fn test_pauli_z_squared_is_identity() {
        let sz = pauli_z();
        let sz2 = mul_2x2(&sz, &sz);
        assert_approx(sz2[0], 1.0, 1e-15, "σ_z²");
        assert_approx(sz2[3], 1.0, 1e-15, "σ_z²");
    }

    #[test]
    fn test_spin_up_eigenstate_of_pauli_z() {
        let result = apply_pauli_z(spin_up());
        assert_approx(result.0, 1.0, 1e-15, "σ_z |↑⟩ = +|↑⟩ (component 0)");
        assert_approx(result.1, 0.0, 1e-15, "σ_z |↑⟩ = +|↑⟩ (component 1)");
    }

    #[test]
    fn test_spin_down_eigenstate_of_pauli_z() {
        // σ_z |↓⟩ = −|↓⟩ = (0, −1)
        let result = apply_pauli_z(spin_down());
        assert_approx(result.0, 0.0, 1e-15, "σ_z |↓⟩ component 0");
        assert_approx(result.1, -1.0, 1e-15, "σ_z |↓⟩ = −|↓⟩ component 1");
    }

    #[test]
    fn test_pauli_x_flips_spin() {
        let result = apply_pauli_x(spin_up());
        assert_approx(result.0, 0.0, 1e-15, "σ_x |↑⟩ = |↓⟩");
        assert_approx(result.1, 1.0, 1e-15, "σ_x |↑⟩ = |↓⟩");
    }

    #[test]
    fn test_pauli_z_trace_zero() {
        assert_approx(trace_2x2(&pauli_z()), 0.0, 1e-15, "Tr(σ_z) = 0");
    }

    #[test]
    fn test_pauli_z_determinant() {
        assert_approx(det_2x2(&pauli_z()), -1.0, 1e-15, "det(σ_z) = -1");
    }

    // --- Density matrix ---

    #[test]
    fn test_density_matrix_pure_spin_up() {
        let rho = density_matrix_pure(spin_up());
        assert_approx(rho[0], 1.0, 1e-15, "ρ₁₁ = 1");
        assert_approx(rho[3], 0.0, 1e-15, "ρ₂₂ = 0");
    }

    #[test]
    fn test_pure_state_detected() {
        let rho = density_matrix_pure(spin_up());
        assert!(is_pure_state(&rho), "spin-up should be pure");
    }

    #[test]
    fn test_mixed_state_detected() {
        // Maximally mixed: ρ = I/2
        let rho = [0.5, 0.0, 0.0, 0.5];
        assert!(is_mixed_state(&rho), "I/2 should be mixed");
        assert!(!is_pure_state(&rho));
    }

    #[test]
    fn test_purity_pure_state() {
        let rho = density_matrix_pure(spin_up());
        assert_approx(purity(&rho), 1.0, 1e-10, "purity of pure state = 1");
    }

    #[test]
    fn test_purity_maximally_mixed() {
        let rho = [0.5, 0.0, 0.0, 0.5];
        assert_approx(purity(&rho), 0.5, 1e-10, "purity of I/2 = 0.5");
    }

    #[test]
    fn test_von_neumann_entropy_pure_is_zero() {
        let rho = density_matrix_pure(spin_up());
        let s = von_neumann_entropy(&rho);
        assert_approx(s, 0.0, 1e-10, "S = 0 for pure state");
    }

    #[test]
    fn test_von_neumann_entropy_mixed_positive() {
        let rho = [0.5, 0.0, 0.0, 0.5];
        let s = von_neumann_entropy(&rho);
        assert!(s > 0.0, "entropy of mixed state > 0");
        // Maximally mixed in 2D: S = ln(2)
        assert_approx(s, 2.0_f64.ln(), 1e-10, "S = ln(2) for maximally mixed");
    }
}
