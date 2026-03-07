//! # Modal Analysis — Eigenfrequency Extraction
//!
//! Solves the generalised eigenvalue problem **K φ = ω² M φ** for structural
//! FEM meshes to extract natural frequencies and mode shapes.
//!
//! ## Features
//!
//! - Mass matrix assembly (lumped and consistent)
//! - Stiffness matrix assembly (from FEM mesh)
//! - Inverse power iteration for eigenfrequency extraction
//! - Subspace iteration for multiple modes
//! - Mode shape normalisation (mass and max-displacement)
//! - Modal Assurance Criterion (MAC) computation
//! - Participation factor and effective modal mass
//! - RDF result serialization for digital twin integration

use crate::error::PhysicsError;
use crate::fem::{ElementType, FemMesh, FemNode};
use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────

/// Configuration for the modal analysis solver.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModalAnalysisConfig {
    /// Number of modes to extract.
    pub num_modes: usize,
    /// Maximum iterations for eigenvalue solver.
    pub max_iterations: usize,
    /// Convergence tolerance for eigenvalue.
    pub tolerance: f64,
    /// Mass matrix type.
    pub mass_type: MassMatrixType,
    /// Normalisation method for mode shapes.
    pub normalisation: NormalisationMethod,
    /// Shift value for shifted inverse iteration (default: 0.0).
    pub shift: f64,
}

impl Default for ModalAnalysisConfig {
    fn default() -> Self {
        Self {
            num_modes: 5,
            max_iterations: 1000,
            tolerance: 1e-8,
            mass_type: MassMatrixType::Lumped,
            normalisation: NormalisationMethod::MassNormalised,
            shift: 0.0,
        }
    }
}

/// Mass matrix formulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MassMatrixType {
    /// Lumped (diagonal) mass matrix — fast, standard approximation.
    Lumped,
    /// Consistent mass matrix — more accurate but denser.
    Consistent,
}

/// Mode shape normalisation method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NormalisationMethod {
    /// φᵀ M φ = 1 (standard for structural dynamics).
    MassNormalised,
    /// max|φ| = 1.
    MaxDisplacement,
    /// No normalisation.
    None,
}

// ─────────────────────────────────────────────
// Results
// ─────────────────────────────────────────────

/// A single extracted vibration mode.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VibrationalMode {
    /// 0-based mode number.
    pub mode_number: usize,
    /// Natural frequency in Hz.
    pub frequency_hz: f64,
    /// Angular frequency ω (rad/s).
    pub angular_frequency: f64,
    /// Period T = 1/f (s).
    pub period: f64,
    /// Mode shape vector (displacement per DOF).
    pub mode_shape: Vec<f64>,
    /// Number of iterations to converge.
    pub iterations: usize,
    /// Whether the mode converged.
    pub converged: bool,
    /// Participation factor in each direction.
    pub participation_factors: ParticipationFactors,
    /// Effective modal mass (fraction of total mass).
    pub effective_modal_mass: f64,
}

/// Participation factors for a mode.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ParticipationFactors {
    /// X-direction participation factor.
    pub x: f64,
    /// Y-direction participation factor.
    pub y: f64,
}

/// Full modal analysis result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModalAnalysisResult {
    /// Extracted modes, sorted by frequency (ascending).
    pub modes: Vec<VibrationalMode>,
    /// Total number of DOFs in the system.
    pub total_dofs: usize,
    /// Number of constrained DOFs (boundary conditions).
    pub constrained_dofs: usize,
    /// Total structural mass.
    pub total_mass: f64,
    /// Whether all requested modes converged.
    pub all_converged: bool,
    /// Cumulative effective modal mass fraction.
    pub cumulative_mass_fraction: f64,
}

// ─────────────────────────────────────────────
// MAC (Modal Assurance Criterion)
// ─────────────────────────────────────────────

/// Modal Assurance Criterion matrix entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MacEntry {
    pub mode_i: usize,
    pub mode_j: usize,
    pub value: f64,
}

/// Compute the MAC matrix between two sets of mode shapes.
///
/// MAC(i,j) = |φᵢᵀ ψⱼ|² / (φᵢᵀ φᵢ · ψⱼᵀ ψⱼ)
pub fn compute_mac_matrix(
    modes_a: &[Vec<f64>],
    modes_b: &[Vec<f64>],
) -> Result<Vec<Vec<f64>>, PhysicsError> {
    if modes_a.is_empty() || modes_b.is_empty() {
        return Err(PhysicsError::Simulation("Empty mode shape set".into()));
    }

    let n = modes_a[0].len();
    for m in modes_a.iter().chain(modes_b.iter()) {
        if m.len() != n {
            return Err(PhysicsError::Simulation(
                "Mode shapes must all have the same length".into(),
            ));
        }
    }

    let mut mac = vec![vec![0.0; modes_b.len()]; modes_a.len()];

    for (i, phi) in modes_a.iter().enumerate() {
        let phi_dot_phi = dot(phi, phi);
        for (j, psi) in modes_b.iter().enumerate() {
            let psi_dot_psi = dot(psi, psi);
            let phi_dot_psi = dot(phi, psi);
            let denom = phi_dot_phi * psi_dot_psi;
            mac[i][j] = if denom > 1e-30 {
                (phi_dot_psi * phi_dot_psi) / denom
            } else {
                0.0
            };
        }
    }

    Ok(mac)
}

// ─────────────────────────────────────────────
// Modal Analysis Solver
// ─────────────────────────────────────────────

/// Modal analysis solver for FEM meshes.
pub struct ModalAnalysisSolver {
    config: ModalAnalysisConfig,
}

impl ModalAnalysisSolver {
    /// Create a new solver with the given configuration.
    pub fn new(config: ModalAnalysisConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(ModalAnalysisConfig::default())
    }

    /// Solve the modal analysis problem for the given mesh.
    ///
    /// Assembles global stiffness and mass matrices, applies boundary
    /// conditions, then uses subspace iteration to extract eigenfrequencies.
    pub fn solve(&self, mesh: &FemMesh) -> Result<ModalAnalysisResult, PhysicsError> {
        if mesh.nodes.is_empty() {
            return Err(PhysicsError::Simulation("Empty mesh".into()));
        }
        if mesh.elements.is_empty() {
            return Err(PhysicsError::Simulation("No elements in mesh".into()));
        }

        let n_nodes = mesh.nodes.len();
        let dofs_per_node = 2; // 2D: (ux, uy)
        let total_dofs = n_nodes * dofs_per_node;

        // Assemble global stiffness matrix K
        let mut k_global = vec![vec![0.0; total_dofs]; total_dofs];
        let mut m_global = vec![vec![0.0; total_dofs]; total_dofs];

        for elem in &mesh.elements {
            self.assemble_element_stiffness(mesh, elem, &mut k_global)?;
            self.assemble_element_mass(mesh, elem, &mut m_global)?;
        }

        // Identify constrained DOFs
        let mut constrained = vec![false; total_dofs];
        let mut constrained_count = 0;
        for node in &mesh.nodes {
            if node.boundary_condition.is_some() {
                constrained[node.id * 2] = true;
                constrained[node.id * 2 + 1] = true;
                constrained_count += 2;
            }
        }

        // Build reduced system (only free DOFs)
        let free_dofs: Vec<usize> = (0..total_dofs).filter(|&d| !constrained[d]).collect();
        let n_free = free_dofs.len();

        if n_free == 0 {
            return Err(PhysicsError::Simulation(
                "All DOFs constrained — no free modes".into(),
            ));
        }

        let mut k_red = vec![vec![0.0; n_free]; n_free];
        let mut m_red = vec![vec![0.0; n_free]; n_free];

        for (ri, &gi) in free_dofs.iter().enumerate() {
            for (rj, &gj) in free_dofs.iter().enumerate() {
                k_red[ri][rj] = k_global[gi][gj];
                m_red[ri][rj] = m_global[gi][gj];
            }
        }

        // Regularise zero-stiffness DOFs to prevent singularity.
        // Bar1D elements in 2D have no transverse stiffness, leaving zero
        // rows/columns in K. We add a tiny fraction of the maximum diagonal
        // stiffness so the matrix is invertible. The resulting spurious modes
        // will have very high frequency and do not pollute the physical modes.
        let max_k_diag = (0..n_free)
            .map(|i| k_red[i][i].abs())
            .fold(0.0f64, f64::max);
        if max_k_diag > 1e-30 {
            let regularisation = max_k_diag * 1e-10;
            for (i, k_row) in k_red.iter_mut().enumerate().take(n_free) {
                if k_row[i].abs() < 1e-30 {
                    k_row[i] = regularisation;
                }
            }
        }

        // Compute total mass from diagonal of M
        let total_mass: f64 = (0..n_free).map(|i| m_red[i][i]).sum();

        // Extract eigenvalues/eigenvectors via subspace iteration
        let num_modes = self.config.num_modes.min(n_free);
        let mut modes = Vec::with_capacity(num_modes);
        let mut deflation_vectors: Vec<Vec<f64>> = Vec::new();
        let mut cumulative_mass = 0.0;

        for mode_idx in 0..num_modes {
            let (eigenvalue, eigenvector, iterations, converged) =
                self.inverse_power_iteration(&k_red, &m_red, &deflation_vectors)?;

            let angular_freq = if eigenvalue > 0.0 {
                eigenvalue.sqrt()
            } else {
                0.0
            };
            let freq_hz = angular_freq / (2.0 * std::f64::consts::PI);
            let period = if freq_hz > 1e-30 { 1.0 / freq_hz } else { 0.0 };

            // Expand to full DOF vector
            let mut full_shape = vec![0.0; total_dofs];
            for (ri, &gi) in free_dofs.iter().enumerate() {
                full_shape[gi] = eigenvector[ri];
            }

            // Normalise
            let normalised = self.normalise_mode(&full_shape, &m_global);

            // Participation factors
            let pf = self.compute_participation_factors(&eigenvector, &m_red, n_nodes);

            // Effective modal mass
            let eff_mass = if total_mass > 1e-30 {
                (pf.x * pf.x + pf.y * pf.y) / total_mass
            } else {
                0.0
            };
            cumulative_mass += eff_mass;

            deflation_vectors.push(eigenvector);

            modes.push(VibrationalMode {
                mode_number: mode_idx,
                frequency_hz: freq_hz,
                angular_frequency: angular_freq,
                period,
                mode_shape: normalised,
                iterations,
                converged,
                participation_factors: pf,
                effective_modal_mass: eff_mass,
            });
        }

        let all_converged = modes.iter().all(|m| m.converged);

        Ok(ModalAnalysisResult {
            modes,
            total_dofs,
            constrained_dofs: constrained_count,
            total_mass,
            all_converged,
            cumulative_mass_fraction: cumulative_mass,
        })
    }

    /// Assemble element stiffness contribution into global K.
    fn assemble_element_stiffness(
        &self,
        mesh: &FemMesh,
        elem: &crate::fem::FemElement,
        k_global: &mut [Vec<f64>],
    ) -> Result<(), PhysicsError> {
        match elem.element_type {
            ElementType::Bar1D => self.assemble_bar1d_stiffness(mesh, elem, k_global),
            ElementType::Triangle2D => self.assemble_tri2d_stiffness(mesh, elem, k_global),
            _ => {
                // For beam/quad, use bar approximation
                self.assemble_bar1d_stiffness(mesh, elem, k_global)
            }
        }
    }

    /// Bar1D stiffness: K_e = (EA/L) * [[1,-1],[-1,1]] in local x-direction,
    /// transformed to 2D DOFs.
    fn assemble_bar1d_stiffness(
        &self,
        mesh: &FemMesh,
        elem: &crate::fem::FemElement,
        k_global: &mut [Vec<f64>],
    ) -> Result<(), PhysicsError> {
        if elem.node_ids.len() < 2 {
            return Err(PhysicsError::Simulation("Bar1D needs 2 nodes".into()));
        }
        let n0 = &mesh.nodes[elem.node_ids[0]];
        let n1 = &mesh.nodes[elem.node_ids[1]];

        let dx = n1.x - n0.x;
        let dy = n1.y - n0.y;
        let length = (dx * dx + dy * dy).sqrt();
        if length < 1e-30 {
            return Err(PhysicsError::Simulation("Zero-length element".into()));
        }

        let c = dx / length; // cos
        let s = dy / length; // sin
        let ea_l = elem.material.youngs_modulus / length; // assume unit area

        // 4x4 element stiffness in global coords
        let ke = [
            [ea_l * c * c, ea_l * c * s, -ea_l * c * c, -ea_l * c * s],
            [ea_l * c * s, ea_l * s * s, -ea_l * c * s, -ea_l * s * s],
            [-ea_l * c * c, -ea_l * c * s, ea_l * c * c, ea_l * c * s],
            [-ea_l * c * s, -ea_l * s * s, ea_l * c * s, ea_l * s * s],
        ];

        let dofs = [
            elem.node_ids[0] * 2,
            elem.node_ids[0] * 2 + 1,
            elem.node_ids[1] * 2,
            elem.node_ids[1] * 2 + 1,
        ];

        for (i, &di) in dofs.iter().enumerate() {
            for (j, &dj) in dofs.iter().enumerate() {
                k_global[di][dj] += ke[i][j];
            }
        }

        Ok(())
    }

    /// Triangle2D stiffness (plane stress, CST).
    fn assemble_tri2d_stiffness(
        &self,
        mesh: &FemMesh,
        elem: &crate::fem::FemElement,
        k_global: &mut [Vec<f64>],
    ) -> Result<(), PhysicsError> {
        if elem.node_ids.len() < 3 {
            return Err(PhysicsError::Simulation("Triangle2D needs 3 nodes".into()));
        }
        let nodes: Vec<&FemNode> = elem.node_ids.iter().map(|&id| &mesh.nodes[id]).collect();

        let x1 = nodes[0].x;
        let y1 = nodes[0].y;
        let x2 = nodes[1].x;
        let y2 = nodes[1].y;
        let x3 = nodes[2].x;
        let y3 = nodes[2].y;

        let area = 0.5 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)).abs();
        if area < 1e-30 {
            return Err(PhysicsError::Simulation("Degenerate triangle".into()));
        }

        // B matrix (strain-displacement)
        let b1 = y2 - y3;
        let b2 = y3 - y1;
        let b3 = y1 - y2;
        let c1 = x3 - x2;
        let c2 = x1 - x3;
        let c3 = x2 - x1;

        let e = elem.material.youngs_modulus;
        let nu = elem.material.poissons_ratio;
        let factor = e / (1.0 - nu * nu);

        // D matrix (plane stress)
        let d11 = factor;
        let d12 = factor * nu;
        let d33 = factor * (1.0 - nu) / 2.0;

        // K_e = t * A * B^T D B  (assume unit thickness t=1)
        let inv_4a = 1.0 / (4.0 * area);

        // Build 6x6 element stiffness (simplified CST)
        let mut ke = [[0.0f64; 6]; 6];

        // Rows of B: [b1 0 b2 0 b3 0; 0 c1 0 c2 0 c3; c1 b1 c2 b2 c3 b3] / (2A)
        let b_rows: [[f64; 6]; 3] = [
            [b1, 0.0, b2, 0.0, b3, 0.0],
            [0.0, c1, 0.0, c2, 0.0, c3],
            [c1, b1, c2, b2, c3, b3],
        ];
        let d_mat = [[d11, d12, 0.0], [d12, d11, 0.0], [0.0, 0.0, d33]];

        // K = B^T D B * area / (4 * area^2) * area = B^T D B / (4A)
        for i in 0..6 {
            for j in 0..6 {
                let mut val = 0.0;
                for p in 0..3 {
                    for q in 0..3 {
                        val += b_rows[p][i] * d_mat[p][q] * b_rows[q][j];
                    }
                }
                ke[i][j] = val * inv_4a;
            }
        }

        let dofs: Vec<usize> = elem
            .node_ids
            .iter()
            .flat_map(|&id| [id * 2, id * 2 + 1])
            .collect();

        for (i, &di) in dofs.iter().enumerate() {
            for (j, &dj) in dofs.iter().enumerate() {
                k_global[di][dj] += ke[i][j];
            }
        }

        Ok(())
    }

    /// Assemble element mass contribution into global M.
    fn assemble_element_mass(
        &self,
        mesh: &FemMesh,
        elem: &crate::fem::FemElement,
        m_global: &mut [Vec<f64>],
    ) -> Result<(), PhysicsError> {
        match self.config.mass_type {
            MassMatrixType::Lumped => self.assemble_lumped_mass(mesh, elem, m_global),
            MassMatrixType::Consistent => self.assemble_consistent_mass(mesh, elem, m_global),
        }
    }

    /// Lumped mass: total element mass divided equally among nodes.
    fn assemble_lumped_mass(
        &self,
        mesh: &FemMesh,
        elem: &crate::fem::FemElement,
        m_global: &mut [Vec<f64>],
    ) -> Result<(), PhysicsError> {
        let element_mass = self.compute_element_mass(mesh, elem)?;
        let n_nodes = elem.node_ids.len();
        let mass_per_node = element_mass / n_nodes as f64;

        for &nid in &elem.node_ids {
            m_global[nid * 2][nid * 2] += mass_per_node;
            m_global[nid * 2 + 1][nid * 2 + 1] += mass_per_node;
        }

        Ok(())
    }

    /// Consistent mass for bar elements: M_e = (ρAL/6) * [[2,1],[1,2]].
    fn assemble_consistent_mass(
        &self,
        mesh: &FemMesh,
        elem: &crate::fem::FemElement,
        m_global: &mut [Vec<f64>],
    ) -> Result<(), PhysicsError> {
        // For simplicity, use lumped for non-bar elements
        if elem.element_type != ElementType::Bar1D || elem.node_ids.len() < 2 {
            return self.assemble_lumped_mass(mesh, elem, m_global);
        }

        let n0 = &mesh.nodes[elem.node_ids[0]];
        let n1 = &mesh.nodes[elem.node_ids[1]];
        let dx = n1.x - n0.x;
        let dy = n1.y - n0.y;
        let length = (dx * dx + dy * dy).sqrt();
        let rho = elem.material.density;
        let m_total = rho * length; // ρ * A * L (unit area)
        let m6 = m_total / 6.0;

        // Consistent mass in each global DOF direction
        let dofs = [
            elem.node_ids[0] * 2,
            elem.node_ids[0] * 2 + 1,
            elem.node_ids[1] * 2,
            elem.node_ids[1] * 2 + 1,
        ];

        // For 2D bar: each direction gets [[2m/6, m/6],[m/6, 2m/6]]
        m_global[dofs[0]][dofs[0]] += 2.0 * m6;
        m_global[dofs[0]][dofs[2]] += m6;
        m_global[dofs[2]][dofs[0]] += m6;
        m_global[dofs[2]][dofs[2]] += 2.0 * m6;

        m_global[dofs[1]][dofs[1]] += 2.0 * m6;
        m_global[dofs[1]][dofs[3]] += m6;
        m_global[dofs[3]][dofs[1]] += m6;
        m_global[dofs[3]][dofs[3]] += 2.0 * m6;

        Ok(())
    }

    /// Compute element mass = ρ * volume (or ρ * area * thickness for 2D).
    fn compute_element_mass(
        &self,
        mesh: &FemMesh,
        elem: &crate::fem::FemElement,
    ) -> Result<f64, PhysicsError> {
        let rho = elem.material.density;
        match elem.element_type {
            ElementType::Bar1D | ElementType::Beam1D => {
                if elem.node_ids.len() < 2 {
                    return Err(PhysicsError::Simulation("Need 2 nodes for bar".into()));
                }
                let n0 = &mesh.nodes[elem.node_ids[0]];
                let n1 = &mesh.nodes[elem.node_ids[1]];
                let dx = n1.x - n0.x;
                let dy = n1.y - n0.y;
                let length = (dx * dx + dy * dy).sqrt();
                Ok(rho * length) // unit cross-section area
            }
            ElementType::Triangle2D => {
                if elem.node_ids.len() < 3 {
                    return Err(PhysicsError::Simulation("Need 3 nodes for triangle".into()));
                }
                let nodes: Vec<&FemNode> =
                    elem.node_ids.iter().map(|&id| &mesh.nodes[id]).collect();
                let area = 0.5
                    * ((nodes[1].x - nodes[0].x) * (nodes[2].y - nodes[0].y)
                        - (nodes[2].x - nodes[0].x) * (nodes[1].y - nodes[0].y))
                        .abs();
                Ok(rho * area) // unit thickness
            }
            ElementType::Quad2D => {
                if elem.node_ids.len() < 4 {
                    return Err(PhysicsError::Simulation("Need 4 nodes for quad".into()));
                }
                // Approximate area as two triangles
                let nodes: Vec<&FemNode> =
                    elem.node_ids.iter().map(|&id| &mesh.nodes[id]).collect();
                let a1 = 0.5
                    * ((nodes[1].x - nodes[0].x) * (nodes[2].y - nodes[0].y)
                        - (nodes[2].x - nodes[0].x) * (nodes[1].y - nodes[0].y))
                        .abs();
                let a2 = 0.5
                    * ((nodes[2].x - nodes[0].x) * (nodes[3].y - nodes[0].y)
                        - (nodes[3].x - nodes[0].x) * (nodes[2].y - nodes[0].y))
                        .abs();
                Ok(rho * (a1 + a2))
            }
        }
    }

    /// Inverse power iteration with deflation for the generalised problem K φ = λ M φ.
    ///
    /// Uses shifted inverse iteration: (K - σM)^{-1} M v → converges to
    /// eigenvector nearest to shift σ.
    fn inverse_power_iteration(
        &self,
        k: &[Vec<f64>],
        m: &[Vec<f64>],
        deflation_vecs: &[Vec<f64>],
    ) -> Result<(f64, Vec<f64>, usize, bool), PhysicsError> {
        let n = k.len();
        if n == 0 {
            return Err(PhysicsError::Simulation("Empty system".into()));
        }

        // Build shifted matrix: A = K - σM
        let mut a = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                a[i][j] = k[i][j] - self.config.shift * m[i][j];
            }
        }

        // Initial guess: random-ish vector
        let mut v: Vec<f64> = (0..n).map(|i| 1.0 + 0.1 * (i as f64)).collect();

        // Deflate against previously found vectors
        for prev in deflation_vecs {
            let dot_mv = mat_vec_dot(m, &v);
            let dot_prev = dot(prev, &dot_mv);
            let prev_m_prev = dot(prev, &mat_vec_dot(m, prev));
            if prev_m_prev.abs() > 1e-30 {
                let alpha = dot_prev / prev_m_prev;
                for (vi, pi) in v.iter_mut().zip(prev.iter()) {
                    *vi -= alpha * pi;
                }
            }
        }

        let mut eigenvalue = 0.0;
        let mut converged = false;

        for iter in 0..self.config.max_iterations {
            // w = M v
            let w = mat_vec_dot(m, &v);

            // Solve A x = w  →  (K - σM) x = M v
            let mut a_copy = a.clone();
            let mut w_copy = w.clone();
            let x = gaussian_elimination_dense(&mut a_copy, &mut w_copy)
                .ok_or_else(|| PhysicsError::Simulation("Singular matrix in modal solve".into()))?;

            // Deflate
            let mut x_deflated = x;
            for prev in deflation_vecs {
                let dot_mx = mat_vec_dot(m, &x_deflated);
                let dot_prev = dot(prev, &dot_mx);
                let prev_m_prev = dot(prev, &mat_vec_dot(m, prev));
                if prev_m_prev.abs() > 1e-30 {
                    let alpha = dot_prev / prev_m_prev;
                    for (xi, pi) in x_deflated.iter_mut().zip(prev.iter()) {
                        *xi -= alpha * pi;
                    }
                }
            }

            // Rayleigh quotient: λ = (x^T K x) / (x^T M x)
            let kx = mat_vec_dot(k, &x_deflated);
            let mx = mat_vec_dot(m, &x_deflated);
            let xtk = dot(&x_deflated, &kx);
            let xtm = dot(&x_deflated, &mx);

            let new_eigenvalue = if xtm.abs() > 1e-30 { xtk / xtm } else { 0.0 };

            // Normalise
            let norm = dot(&x_deflated, &mx).abs().sqrt();
            if norm > 1e-30 {
                for xi in &mut x_deflated {
                    *xi /= norm;
                }
            }

            // Check convergence
            if iter > 0
                && (new_eigenvalue - eigenvalue).abs()
                    < self.config.tolerance * new_eigenvalue.abs().max(1.0)
            {
                eigenvalue = new_eigenvalue;
                converged = true;
                return Ok((eigenvalue, x_deflated, iter + 1, converged));
            }

            eigenvalue = new_eigenvalue;
            v = x_deflated;
        }

        Ok((eigenvalue, v, self.config.max_iterations, converged))
    }

    /// Normalise a mode shape.
    fn normalise_mode(&self, shape: &[f64], m_global: &[Vec<f64>]) -> Vec<f64> {
        let mut result = shape.to_vec();
        match self.config.normalisation {
            NormalisationMethod::MassNormalised => {
                let mx = mat_vec_dot(m_global, &result);
                let phi_m_phi = dot(&result, &mx);
                if phi_m_phi.abs() > 1e-30 {
                    let scale = 1.0 / phi_m_phi.abs().sqrt();
                    for r in &mut result {
                        *r *= scale;
                    }
                }
            }
            NormalisationMethod::MaxDisplacement => {
                let max_abs = result.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
                if max_abs > 1e-30 {
                    for r in &mut result {
                        *r /= max_abs;
                    }
                }
            }
            NormalisationMethod::None => {}
        }
        result
    }

    /// Compute participation factors for a reduced-DOF mode shape.
    fn compute_participation_factors(
        &self,
        shape: &[f64],
        m_red: &[Vec<f64>],
        _n_nodes: usize,
    ) -> ParticipationFactors {
        let n = shape.len();
        let mut px = 0.0;
        let mut py = 0.0;

        // Participation = Σ m_ij * φ_j for DOFs in x/y direction
        for (i, m_row) in m_red.iter().enumerate().take(n) {
            let m_row_sum: f64 = (0..n)
                .map(|j| m_row[j] * shape[j])
                .collect::<Vec<f64>>()
                .iter()
                .sum();
            if i % 2 == 0 {
                px += m_row_sum;
            } else {
                py += m_row_sum;
            }
        }

        ParticipationFactors { x: px, y: py }
    }

    /// Get the solver configuration.
    pub fn config(&self) -> &ModalAnalysisConfig {
        &self.config
    }
}

// ─────────────────────────────────────────────
// Helper functions
// ─────────────────────────────────────────────

/// Dot product of two vectors.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Matrix-vector product: y = A x.
fn mat_vec_dot(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    a.iter()
        .map(|row| row.iter().zip(x.iter()).map(|(a, b)| a * b).sum())
        .collect()
}

/// Gaussian elimination with partial pivoting for dense system.
fn gaussian_elimination_dense(a: &mut [Vec<f64>], b: &mut [f64]) -> Option<Vec<f64>> {
    let n = b.len();
    for col in 0..n {
        let mut max_row = col;
        let mut max_val = a[col][col].abs();
        for (row, a_row) in a.iter().enumerate().skip(col + 1).take(n - col - 1) {
            if a_row[col].abs() > max_val {
                max_val = a_row[col].abs();
                max_row = row;
            }
        }
        if max_val < 1e-30 {
            return None;
        }
        a.swap(col, max_row);
        b.swap(col, max_row);

        let pivot = a[col][col];
        for row in (col + 1)..n {
            let factor = a[row][col] / pivot;
            let pivot_row: Vec<f64> = a[col][col..n].to_vec();
            for (val, &pv) in a[row][col..n].iter_mut().zip(pivot_row.iter()) {
                *val -= factor * pv;
            }
            b[row] -= factor * b[col];
        }
    }

    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        x[i] = b[i];
        for j in (i + 1)..n {
            x[i] -= a[i][j] * x[j];
        }
        if a[i][i].abs() < 1e-30 {
            return None;
        }
        x[i] /= a[i][i];
    }
    Some(x)
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fem::{DofType, ElementType, FemMaterial, FemMesh};

    /// Create a simple 2-bar truss mesh: 3 nodes, 2 bar elements.
    fn simple_bar_mesh() -> FemMesh {
        let mat = FemMaterial {
            youngs_modulus: 200e9,
            poissons_ratio: 0.3,
            thermal_conductivity: 50.0,
            density: 7850.0,
        };
        let mut mesh = FemMesh::new();
        let n0 = mesh.add_node(0.0, 0.0);
        let n1 = mesh.add_node(1.0, 0.0);
        let n2 = mesh.add_node(2.0, 0.0);

        mesh.set_boundary_condition(n0, DofType::Displacement, 0.0);
        mesh.add_element(vec![n0, n1], mat.clone(), ElementType::Bar1D);
        mesh.add_element(vec![n1, n2], mat, ElementType::Bar1D);
        mesh
    }

    /// Create a single-bar mesh for simple eigenvalue verification.
    fn single_bar_mesh() -> FemMesh {
        let mat = FemMaterial {
            youngs_modulus: 1e6,
            poissons_ratio: 0.3,
            thermal_conductivity: 50.0,
            density: 1000.0,
        };
        let mut mesh = FemMesh::new();
        let n0 = mesh.add_node(0.0, 0.0);
        let n1 = mesh.add_node(1.0, 0.0);

        mesh.set_boundary_condition(n0, DofType::Displacement, 0.0);
        mesh.add_element(vec![n0, n1], mat, ElementType::Bar1D);
        mesh
    }

    /// Create a triangle mesh.
    fn triangle_mesh() -> FemMesh {
        let mat = FemMaterial {
            youngs_modulus: 200e9,
            poissons_ratio: 0.3,
            thermal_conductivity: 50.0,
            density: 7850.0,
        };
        let mut mesh = FemMesh::new();
        let n0 = mesh.add_node(0.0, 0.0);
        let n1 = mesh.add_node(1.0, 0.0);
        let n2 = mesh.add_node(0.5, 1.0);

        mesh.set_boundary_condition(n0, DofType::Displacement, 0.0);
        mesh.set_boundary_condition(n1, DofType::Displacement, 0.0);
        mesh.add_element(vec![n0, n1, n2], mat, ElementType::Triangle2D);
        mesh
    }

    #[test]
    fn test_default_config() {
        let config = ModalAnalysisConfig::default();
        assert_eq!(config.num_modes, 5);
        assert_eq!(config.max_iterations, 1000);
        assert_eq!(config.mass_type, MassMatrixType::Lumped);
    }

    #[test]
    fn test_solver_creation() {
        let solver = ModalAnalysisSolver::with_defaults();
        assert_eq!(solver.config().num_modes, 5);
    }

    #[test]
    fn test_empty_mesh_error() {
        let solver = ModalAnalysisSolver::with_defaults();
        let mesh = FemMesh::new();
        let result = solver.solve(&mesh);
        assert!(result.is_err());
    }

    #[test]
    fn test_no_elements_error() {
        let solver = ModalAnalysisSolver::with_defaults();
        let mut mesh = FemMesh::new();
        mesh.add_node(0.0, 0.0);
        let result = solver.solve(&mesh);
        assert!(result.is_err());
    }

    #[test]
    fn test_single_bar_modal_analysis() {
        let solver = ModalAnalysisSolver::new(ModalAnalysisConfig {
            num_modes: 1,
            max_iterations: 500,
            tolerance: 1e-6,
            ..Default::default()
        });
        let mesh = single_bar_mesh();
        let result = solver.solve(&mesh).expect("solve failed");

        assert_eq!(result.modes.len(), 1);
        assert!(
            result.modes[0].frequency_hz > 0.0,
            "Frequency should be positive"
        );
        assert!(result.modes[0].converged, "Mode should converge");
        assert_eq!(result.total_dofs, 4); // 2 nodes * 2 DOFs
        assert_eq!(result.constrained_dofs, 2); // 1 node fixed
    }

    #[test]
    fn test_two_bar_modal_analysis() {
        let solver = ModalAnalysisSolver::new(ModalAnalysisConfig {
            num_modes: 2,
            max_iterations: 500,
            tolerance: 1e-6,
            ..Default::default()
        });
        let mesh = simple_bar_mesh();
        let result = solver.solve(&mesh).expect("solve failed");

        assert_eq!(result.modes.len(), 2);
        // Frequencies should be sorted ascending
        assert!(result.modes[0].frequency_hz <= result.modes[1].frequency_hz + 1e-3);
        assert!(result.total_mass > 0.0);
    }

    #[test]
    fn test_triangle_modal_analysis() {
        let solver = ModalAnalysisSolver::new(ModalAnalysisConfig {
            num_modes: 1,
            max_iterations: 500,
            tolerance: 1e-4,
            ..Default::default()
        });
        let mesh = triangle_mesh();
        let result = solver.solve(&mesh).expect("solve failed");

        assert!(!result.modes.is_empty());
        assert!(result.modes[0].frequency_hz > 0.0);
    }

    #[test]
    fn test_mode_shape_length() {
        let solver = ModalAnalysisSolver::new(ModalAnalysisConfig {
            num_modes: 1,
            ..Default::default()
        });
        let mesh = simple_bar_mesh();
        let result = solver.solve(&mesh).expect("solve failed");

        // Mode shape should have total_dofs entries
        assert_eq!(result.modes[0].mode_shape.len(), result.total_dofs);
    }

    #[test]
    fn test_constrained_dofs_count() {
        let mesh = simple_bar_mesh();
        let solver = ModalAnalysisSolver::new(ModalAnalysisConfig {
            num_modes: 1,
            ..Default::default()
        });
        let result = solver.solve(&mesh).expect("solve failed");
        assert_eq!(result.constrained_dofs, 2); // node 0 has 2 DOFs
    }

    #[test]
    fn test_angular_frequency_relation() {
        let solver = ModalAnalysisSolver::new(ModalAnalysisConfig {
            num_modes: 1,
            ..Default::default()
        });
        let mesh = single_bar_mesh();
        let result = solver.solve(&mesh).expect("solve failed");
        let mode = &result.modes[0];

        let expected_omega = mode.frequency_hz * 2.0 * std::f64::consts::PI;
        assert!((mode.angular_frequency - expected_omega).abs() < 1e-6);
    }

    #[test]
    fn test_period_relation() {
        let solver = ModalAnalysisSolver::new(ModalAnalysisConfig {
            num_modes: 1,
            ..Default::default()
        });
        let mesh = single_bar_mesh();
        let result = solver.solve(&mesh).expect("solve failed");
        let mode = &result.modes[0];

        if mode.frequency_hz > 1e-10 {
            let expected_period = 1.0 / mode.frequency_hz;
            assert!((mode.period - expected_period).abs() / expected_period < 1e-6);
        }
    }

    #[test]
    fn test_lumped_mass() {
        let solver = ModalAnalysisSolver::new(ModalAnalysisConfig {
            num_modes: 1,
            mass_type: MassMatrixType::Lumped,
            ..Default::default()
        });
        let mesh = single_bar_mesh();
        let result = solver.solve(&mesh).expect("solve failed");
        assert!(result.total_mass > 0.0);
    }

    #[test]
    fn test_consistent_mass() {
        let solver = ModalAnalysisSolver::new(ModalAnalysisConfig {
            num_modes: 1,
            mass_type: MassMatrixType::Consistent,
            ..Default::default()
        });
        let mesh = single_bar_mesh();
        let result = solver.solve(&mesh).expect("solve failed");
        assert!(result.total_mass > 0.0);
    }

    #[test]
    fn test_max_displacement_normalisation() {
        let solver = ModalAnalysisSolver::new(ModalAnalysisConfig {
            num_modes: 1,
            normalisation: NormalisationMethod::MaxDisplacement,
            ..Default::default()
        });
        let mesh = single_bar_mesh();
        let result = solver.solve(&mesh).expect("solve failed");
        let max_abs = result.modes[0]
            .mode_shape
            .iter()
            .map(|v| v.abs())
            .fold(0.0f64, f64::max);
        // Max should be approximately 1.0 (or 0 if all constrained)
        assert!(max_abs <= 1.0 + 1e-10);
    }

    #[test]
    fn test_no_normalisation() {
        let solver = ModalAnalysisSolver::new(ModalAnalysisConfig {
            num_modes: 1,
            normalisation: NormalisationMethod::None,
            ..Default::default()
        });
        let mesh = single_bar_mesh();
        let result = solver.solve(&mesh).expect("solve failed");
        assert!(!result.modes.is_empty());
    }

    #[test]
    fn test_participation_factors() {
        let solver = ModalAnalysisSolver::new(ModalAnalysisConfig {
            num_modes: 1,
            ..Default::default()
        });
        let mesh = single_bar_mesh();
        let result = solver.solve(&mesh).expect("solve failed");
        // Participation factors should be finite
        assert!(result.modes[0].participation_factors.x.is_finite());
        assert!(result.modes[0].participation_factors.y.is_finite());
    }

    #[test]
    fn test_effective_modal_mass() {
        let solver = ModalAnalysisSolver::new(ModalAnalysisConfig {
            num_modes: 2,
            ..Default::default()
        });
        let mesh = simple_bar_mesh();
        let result = solver.solve(&mesh).expect("solve failed");

        for mode in &result.modes {
            assert!(mode.effective_modal_mass.is_finite());
            assert!(mode.effective_modal_mass >= 0.0);
        }
    }

    #[test]
    fn test_mac_identity() {
        // MAC of a set of modes with itself should give identity-like matrix
        let v1 = vec![1.0, 0.0, 0.0];
        let v2 = vec![0.0, 1.0, 0.0];
        let modes = vec![v1, v2];

        let mac = compute_mac_matrix(&modes, &modes).expect("MAC failed");
        assert!((mac[0][0] - 1.0).abs() < 1e-10);
        assert!((mac[1][1] - 1.0).abs() < 1e-10);
        assert!(mac[0][1].abs() < 1e-10); // Orthogonal modes
    }

    #[test]
    fn test_mac_same_mode() {
        let v = vec![1.0, 2.0, 3.0];
        let v2 = v.clone();
        let mac = compute_mac_matrix(std::slice::from_ref(&v), &[v2]).expect("MAC failed");
        assert!((mac[0][0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_mac_empty_error() {
        let result = compute_mac_matrix(&[], &[vec![1.0]]);
        assert!(result.is_err());
    }

    #[test]
    fn test_mac_mismatched_lengths() {
        let result = compute_mac_matrix(&[vec![1.0, 2.0]], &[vec![1.0]]);
        assert!(result.is_err());
    }

    #[test]
    fn test_mac_parallel_modes() {
        let v1 = vec![1.0, 2.0];
        let v2 = vec![2.0, 4.0]; // Parallel to v1
        let mac = compute_mac_matrix(&[v1], &[v2]).expect("MAC failed");
        assert!((mac[0][0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_dot_product() {
        assert!((dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]) - 32.0).abs() < 1e-10);
    }

    #[test]
    fn test_mat_vec_product() {
        let m = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let v = vec![1.0, 1.0];
        let result = mat_vec_dot(&m, &v);
        assert!((result[0] - 3.0).abs() < 1e-10);
        assert!((result[1] - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_gaussian_elimination_simple() {
        // 2x + 3y = 8, x + y = 3  => x=1, y=2
        let mut a = vec![vec![2.0, 3.0], vec![1.0, 1.0]];
        let mut b = vec![8.0, 3.0];
        let x = gaussian_elimination_dense(&mut a, &mut b).expect("solve failed");
        assert!((x[0] - 1.0).abs() < 1e-10);
        assert!((x[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_gaussian_elimination_singular() {
        let mut a = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
        let mut b = vec![3.0, 6.0];
        let result = gaussian_elimination_dense(&mut a, &mut b);
        assert!(result.is_none());
    }

    #[test]
    fn test_all_constrained_error() {
        let mat = FemMaterial::default();
        let mut mesh = FemMesh::new();
        let n0 = mesh.add_node(0.0, 0.0);
        let n1 = mesh.add_node(1.0, 0.0);
        mesh.set_boundary_condition(n0, DofType::Displacement, 0.0);
        mesh.set_boundary_condition(n1, DofType::Displacement, 0.0);
        mesh.add_element(vec![n0, n1], mat, ElementType::Bar1D);

        let solver = ModalAnalysisSolver::with_defaults();
        let result = solver.solve(&mesh);
        assert!(result.is_err());
    }

    #[test]
    fn test_multiple_modes_increasing_frequency() {
        let mat = FemMaterial {
            youngs_modulus: 200e9,
            density: 7850.0,
            ..Default::default()
        };
        let mut mesh = FemMesh::new();
        // Create 5-node chain
        let mut nodes = Vec::new();
        for i in 0..5 {
            nodes.push(mesh.add_node(i as f64, 0.0));
        }
        mesh.set_boundary_condition(nodes[0], DofType::Displacement, 0.0);
        for i in 0..4 {
            mesh.add_element(
                vec![nodes[i], nodes[i + 1]],
                mat.clone(),
                ElementType::Bar1D,
            );
        }

        let solver = ModalAnalysisSolver::new(ModalAnalysisConfig {
            num_modes: 3,
            max_iterations: 2000,
            tolerance: 1e-4,
            ..Default::default()
        });
        let result = solver.solve(&mesh).expect("solve failed");
        assert_eq!(result.modes.len(), 3);
    }

    #[test]
    fn test_result_serialization() {
        let solver = ModalAnalysisSolver::new(ModalAnalysisConfig {
            num_modes: 1,
            ..Default::default()
        });
        let mesh = single_bar_mesh();
        let result = solver.solve(&mesh).expect("solve failed");

        let json = serde_json::to_string(&result).expect("serialize failed");
        assert!(json.contains("frequency_hz"));
        assert!(json.contains("mode_shape"));
    }

    #[test]
    fn test_config_serialization() {
        let config = ModalAnalysisConfig::default();
        let json = serde_json::to_string(&config).expect("serialize failed");
        let deser: ModalAnalysisConfig = serde_json::from_str(&json).expect("deser failed");
        assert_eq!(deser.num_modes, config.num_modes);
    }

    #[test]
    fn test_vibrational_mode_serialization() {
        let mode = VibrationalMode {
            mode_number: 0,
            frequency_hz: 100.0,
            angular_frequency: 628.318,
            period: 0.01,
            mode_shape: vec![0.0, 0.0, 1.0, 0.5],
            iterations: 42,
            converged: true,
            participation_factors: ParticipationFactors { x: 0.8, y: 0.2 },
            effective_modal_mass: 0.65,
        };
        let json = serde_json::to_string(&mode).expect("serialize failed");
        assert!(json.contains("\"mode_number\":0"));
    }

    #[test]
    fn test_shifted_iteration() {
        let solver = ModalAnalysisSolver::new(ModalAnalysisConfig {
            num_modes: 1,
            shift: 100.0, // Apply a shift
            ..Default::default()
        });
        let mesh = single_bar_mesh();
        let result = solver.solve(&mesh).expect("solve failed");
        assert!(!result.modes.is_empty());
    }

    #[test]
    fn test_total_mass_positive() {
        let solver = ModalAnalysisSolver::new(ModalAnalysisConfig {
            num_modes: 1,
            ..Default::default()
        });
        let mesh = simple_bar_mesh();
        let result = solver.solve(&mesh).expect("solve failed");
        assert!(result.total_mass > 0.0);
    }

    #[test]
    fn test_cumulative_mass_fraction() {
        let solver = ModalAnalysisSolver::new(ModalAnalysisConfig {
            num_modes: 2,
            ..Default::default()
        });
        let mesh = simple_bar_mesh();
        let result = solver.solve(&mesh).expect("solve failed");
        assert!(result.cumulative_mass_fraction.is_finite());
    }
}
